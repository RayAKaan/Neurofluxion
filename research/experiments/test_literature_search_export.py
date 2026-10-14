"""Offline unit tests for the literature metadata export helpers."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from literature_search_export import (
    deduplicate,
    normalize_crossref,
    normalize_doi,
    normalize_openalex,
    reconstruct_abstract,
    write_exports,
)


class LiteratureSearchExportTests(unittest.TestCase):
    def test_normalize_doi_variants(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1234/ABC "), "10.1234/abc")
        self.assertEqual(normalize_doi("doi:10.1234/ABC"), "10.1234/abc")

    def test_reconstruct_openalex_inverted_abstract(self):
        self.assertEqual(
            reconstruct_abstract({"network": [0, 2], "trace": [1]}),
            "network trace network",
        )

    def test_normalize_openalex_record(self):
        record = normalize_openalex({
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1234/xyz",
            "title": "A Graph Study",
            "publication_year": 2026,
            "authorships": [{"author": {"display_name": "Ada Example"}}],
            "primary_location": {"source": {"display_name": "Test Venue"}},
            "abstract_inverted_index": {"Graph": [0], "study": [1]},
        }, "graph query")
        self.assertEqual(record["doi"], "10.1234/xyz")
        self.assertEqual(record["authors"], "Ada Example")
        self.assertEqual(record["abstract"], "Graph study")

    def test_normalize_crossref_record(self):
        record = normalize_crossref({
            "DOI": "10.1234/xyz",
            "title": ["A Graph Study"],
            "published": {"date-parts": [[2026, 5, 1]]},
            "author": [{"given": "Ada", "family": "Example"}],
            "container-title": ["Test Venue"],
        }, "graph query")
        self.assertEqual(record["year"], 2026)
        self.assertEqual(record["authors"], "Ada Example")

    def test_deduplicate_by_doi_merges_sources_and_queries(self):
        left = normalize_openalex({
            "id": "W1", "doi": "https://doi.org/10.1234/xyz", "title": "A Graph Study",
            "publication_year": 2026, "authorships": [], "primary_location": {},
        }, "query one")
        right = normalize_crossref({
            "DOI": "10.1234/XYZ", "title": ["A Graph Study"],
            "published": {"date-parts": [[2026]]}, "author": [], "container-title": ["Venue"],
        }, "query two")
        result = deduplicate([left, right])
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]["source_databases"]), {"OpenAlex", "Crossref"})
        self.assertEqual(result[0]["matched_queries"], ["query one", "query two"])

    def test_deduplicate_without_doi_uses_normalized_title_and_year(self):
        left = {"title": "Trace: A Study!", "year": 2025, "doi": "", "source_database": "A",
                "matched_queries": ["q1"], "source_ids": ["a"], "source_records": []}
        right = {"title": "TRACE A STUDY", "year": 2025, "doi": "", "source_database": "B",
                 "matched_queries": ["q2"], "source_ids": ["b"], "source_records": []}
        self.assertEqual(len(deduplicate([left, right])), 1)

    def test_exports_include_blank_screening_columns(self):
        record = deduplicate([{
            "title": "A Study", "doi": "10.1234/a", "year": 2025, "venue": "Venue",
            "authors": "A. Example", "abstract": "", "source_database": "OpenAlex",
            "matched_queries": ["query"], "source_ids": ["W1"], "source_records": [],
        }])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_exports(output, record, record, {"status": "test"})
            self.assertTrue((output / "raw_records.json").exists())
            self.assertTrue((output / "deduplicated_records.json").exists())
            self.assertTrue((output / "search_manifest.json").exists())
            with (output / "screening.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["screening_decision"], "")
            self.assertEqual(row["full_text_status"], "")
            json.loads((output / "search_manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
