#!/usr/bin/env python3
"""Export reproducible, deduplicated literature-search records from public APIs.

This is a metadata discovery/export aid, not an automated relevance classifier
or a completed systematic review. Human screening and full-text verification
remain explicit downstream steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.0.0"
USER_AGENT = "Neurofluxion-Literature-Review/1.0 (reproducible metadata export)"
QUERIES = [
    "neural network visualization",
    "interactive neural network visualization",
    "deep learning visual analytics",
    "neural network debugging operation trace",
    "machine learning provenance",
    "execution trace visualization",
]
OPENALEX_SELECT = ",".join([
    "id", "doi", "title", "publication_year", "type", "cited_by_count",
    "authorships", "primary_location", "open_access", "abstract_inverted_index",
])
CROSSREF_SELECT = ",".join([
    "DOI", "title", "published", "author", "container-title", "type",
    "is-referenced-by-count", "URL", "abstract",
])
class SearchPartialFailure(RuntimeError):
    """A search page failed after zero or more earlier pages succeeded."""

    def __init__(self, message: str, records: list[dict[str, Any]], requests: list[dict[str, Any]]):
        super().__init__(message)
        self.records = records
        self.requests = requests


CSV_FIELDS = [
    "record_key", "title", "doi", "year", "venue", "authors", "abstract",
    "source_databases", "matched_queries", "source_ids", "screening_decision",
    "exclusion_reason", "full_text_status", "reviewer_notes",
]


def fetch_json(url: str, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    doi = value.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(" .;,)")


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words: dict[int, str] = {}
    for word, positions in index.items():
        for position in positions:
            words[int(position)] = word
    return " ".join(words[position] for position in sorted(words))


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def normalize_openalex(item: dict[str, Any], query: str) -> dict[str, Any]:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    authors = [
        (authorship.get("author") or {}).get("display_name", "")
        for authorship in item.get("authorships") or []
    ]
    return {
        "record_key": "",
        "title": str(item.get("title") or "").strip(),
        "doi": normalize_doi(item.get("doi")),
        "year": item.get("publication_year") or "",
        "venue": str(source.get("display_name") or ""),
        "authors": "; ".join(name for name in authors if name),
        "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
        "source_database": "OpenAlex",
        "matched_queries": [query],
        "source_ids": [str(item.get("id") or item.get("doi") or item.get("title") or "")],
        "source_records": [{"database": "OpenAlex", "id": item.get("id", ""), "type": item.get("type", "")}],
    }


def normalize_crossref(item: dict[str, Any], query: str) -> dict[str, Any]:
    date_parts = ((item.get("published") or {}).get("date-parts") or [])
    year = date_parts[0][0] if date_parts and date_parts[0] else ""
    authors = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return {
        "record_key": "",
        "title": _first(item.get("title")).strip(),
        "doi": normalize_doi(item.get("DOI")),
        "year": year,
        "venue": _first(item.get("container-title")),
        "authors": "; ".join(authors),
        "abstract": re.sub(r"<[^>]+>", " ", str(item.get("abstract") or "")).strip(),
        "source_database": "Crossref",
        "matched_queries": [query],
        "source_ids": [str(item.get("DOI") or item.get("URL") or item.get("title") or "")],
        "source_records": [{"database": "Crossref", "id": item.get("DOI", ""), "type": item.get("type", "")}],
    }


def _identity(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return "doi:" + doi
    title = re.sub(r"[^a-z0-9]+", " ", str(record.get("title") or "").lower()).strip()
    year = str(record.get("year") or "")
    if title:
        return "title-year:" + title + "|" + year
    stable = json.dumps(record.get("source_ids", []), sort_keys=True)
    return "source:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for original in records:
        record = dict(original)
        key = _identity(record)
        if key not in merged:
            record["record_key"] = key
            record["source_databases"] = sorted(set(record.get("source_databases", []) or [record.get("source_database", "")]) - {""})
            record["matched_queries"] = sorted(set(record.get("matched_queries", [])))
            record["source_ids"] = sorted(set(record.get("source_ids", [])))
            record["source_records"] = list(record.get("source_records", []))
            merged[key] = record
            continue
        current = merged[key]
        for field in ("title", "doi", "year", "venue", "authors", "abstract"):
            if not current.get(field) and record.get(field):
                current[field] = record[field]
        current["source_databases"] = sorted(set(current.get("source_databases", [])) | {record.get("source_database", "")} - {""})
        current["matched_queries"] = sorted(set(current.get("matched_queries", [])) | set(record.get("matched_queries", [])))
        current["source_ids"] = sorted(set(current.get("source_ids", [])) | set(record.get("source_ids", [])))
        current["source_records"].extend(record.get("source_records", []))
    return sorted(merged.values(), key=lambda item: (str(item.get("year") or ""), item.get("title", "").lower()), reverse=True)


def search_openalex(query: str, page_size: int, max_pages: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    cursor = "*"
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    for page in range(max_pages):
        params = urllib.parse.urlencode({
            "search": query, "per-page": page_size, "cursor": cursor, "select": OPENALEX_SELECT,
        })
        manifest_url = "https://api.openalex.org/works?" + params
        request_url = manifest_url
        if api_key:
            request_url += "&api_key=" + urllib.parse.quote(api_key, safe="")
        try:
            payload = fetch_json(request_url)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if api_key:
                error_text = error_text.replace(api_key, "[REDACTED]")
            requests.append({
                "page": page + 1, "url": manifest_url, "records_returned": 0,
                "status": "failed", "error": error_text,
            })
            # Do not chain the original exception: fetch_json's error message
            # includes its request URL, which may contain the API key.
            raise SearchPartialFailure(
                f"OpenAlex page {page + 1} failed: {error_text}",
                records, requests,
            ) from None
        results = payload.get("results") or []
        records.extend(normalize_openalex(item, query) for item in results)
        requests.append({
            "page": page + 1, "url": manifest_url, "records_returned": len(results), "status": "success",
        })
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not results:
            break
        time.sleep(0.15)
    return records, requests


def search_crossref(query: str, page_size: int, max_pages: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for page in range(max_pages):
        params = urllib.parse.urlencode({
            "query.bibliographic": query, "rows": page_size, "offset": page * page_size,
            "select": CROSSREF_SELECT,
        })
        url = "https://api.crossref.org/works?" + params
        try:
            payload = fetch_json(url)
        except Exception as exc:
            requests.append({
                "page": page + 1, "url": url, "records_returned": 0,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
            raise SearchPartialFailure(
                f"Crossref page {page + 1} failed: {type(exc).__name__}: {exc}",
                records, requests,
            ) from exc
        message = payload.get("message") or {}
        items = message.get("items") or []
        records.extend(normalize_crossref(item, query) for item in items)
        requests.append({
            "page": page + 1, "url": url, "records_returned": len(items), "status": "success",
        })
        if len(items) < page_size or (page + 1) * page_size >= int(message.get("total-results") or 0):
            break
        time.sleep(0.15)
    return records, requests


def write_exports(output: Path, raw_records: list[dict[str, Any]], unique_records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_records.json").write_text(json.dumps(raw_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "deduplicated_records.json").write_text(json.dumps(unique_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "search_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "screening.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in unique_records:
            writer.writerow({
                "record_key": record.get("record_key", ""),
                "title": record.get("title", ""),
                "doi": record.get("doi", ""),
                "year": record.get("year", ""),
                "venue": record.get("venue", ""),
                "authors": record.get("authors", ""),
                "abstract": record.get("abstract", ""),
                "source_databases": "; ".join(record.get("source_databases", [])),
                "matched_queries": " || ".join(record.get("matched_queries", [])),
                "source_ids": "; ".join(record.get("source_ids", [])),
                "screening_decision": "",
                "exclusion_reason": "",
                "full_text_status": "",
                "reviewer_notes": "",
            })


def run_search(output: Path, queries: list[str] = QUERIES, page_size: int = 100, max_pages: int = 2) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    query_runs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for query in queries:
        for database, search_fn in (("OpenAlex", search_openalex), ("Crossref", search_crossref)):
            try:
                records, requests = search_fn(query, page_size, max_pages)
                raw.extend(records)
                query_runs.append({
                    "query": query, "database": database,
                    "requests": requests, "records_returned": len(records), "status": "success",
                })
            except SearchPartialFailure as exc:
                # Preserve completed-page records. The previous implementation
                # discarded them when a later page raised.
                raw.extend(exc.records)
                cause = exc.__cause__ or exc
                failure = {
                    "query": query, "database": database,
                    "error": f"{type(cause).__name__}: {cause}",
                    "partial_records_preserved": len(exc.records),
                }
                failures.append(failure)
                query_runs.append({
                    **failure, "requests": exc.requests, "records_returned": len(exc.records),
                    "status": "partial_failed" if exc.records else "failed",
                })
            except Exception as exc:  # preserve failures in the manifest for reproducibility
                failure = {"query": query, "database": database, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                query_runs.append({**failure, "requests": [], "records_returned": 0, "status": "failed"})
    unique = deduplicate(raw)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "databases": ["OpenAlex", "Crossref"],
        "page_size": page_size,
        "max_pages_per_query": max_pages,
        "query_runs": query_runs,
        "raw_record_count": len(raw),
        "deduplicated_record_count": len(unique),
        "deduplication_rule": "normalized DOI; otherwise normalized title + year; otherwise source-ID hash",
        "failures": failures,
        "successful_query_count": sum(run.get("status") == "success" for run in query_runs),
        "partial_failed_query_count": sum(run.get("status") == "partial_failed" for run in query_runs),
        "failed_query_count": sum(run.get("status") == "failed" for run in query_runs),
        "screening_status": "not screened; CSV decision fields intentionally blank",
        "limitation": "API metadata discovery is not a complete systematic review; search coverage, full-text screening, duplicate adjudication, and independent review remain required.",
    }
    write_exports(output, raw, unique, manifest)
    if not raw:
        raise RuntimeError("No records were retrieved from either source; inspect search_manifest.json for failures.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for raw, deduplicated, manifest, and screening exports")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--query", action="append", dest="queries", help="Override default query; repeat for multiple queries")
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 200:
        parser.error("--page-size must be between 1 and 200")
    if args.max_pages < 1 or args.max_pages > 10:
        parser.error("--max-pages must be between 1 and 10")
    manifest = run_search(Path(args.out), args.queries or QUERIES, args.page_size, args.max_pages)
    print(json.dumps({
        "output": str(Path(args.out)),
        "raw_record_count": manifest["raw_record_count"],
        "deduplicated_record_count": manifest["deduplicated_record_count"],
        "query_failures": len(manifest["failures"]),
        "successful_queries": manifest["successful_query_count"],
        "partial_failed_queries": manifest["partial_failed_query_count"],
        "failed_queries": manifest["failed_query_count"],
    }, indent=2))
    # A non-empty artifact is not necessarily a complete search. Preserve/upload
    # its files, but let CI fail when any source/query failed so it cannot look green.
    return 2 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
