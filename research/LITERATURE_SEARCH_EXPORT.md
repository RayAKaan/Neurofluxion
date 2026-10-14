# Reproducible Literature Search Export

## Purpose

The script `research/experiments/literature_search_export.py` queries public OpenAlex and Crossref metadata APIs with a versioned, fixed query list. It exports raw records, deduplicated records, a manifest containing the exact request URLs and counts, and a CSV screening sheet.

## Default query set

1. neural network visualization
2. interactive neural network visualization
3. deep learning visual analytics
4. neural network debugging operation trace
5. machine learning provenance
6. execution trace visualization

Each query is sent to both sources. The default export takes up to two pages of 100 records per query per source. Workflow-dispatch inputs can increase or reduce this cap.

## Outputs

- `raw_records.json`: source-specific records before deduplication.
- `deduplicated_records.json`: records merged by normalized DOI, or normalized title plus year when DOI is missing.
- `search_manifest.json`: timestamp, script version, query list, exact request URLs, counts, failures, and limitations.
- `screening.csv`: reviewer-ready records with blank decision, exclusion-reason, full-text-status, and reviewer-notes columns.

## Reproducibility and review safeguards

- The API query, page size, maximum pages, source, and request URL are recorded.
- Failed queries are preserved in the manifest; a generated artifact is not automatically a successful search.
- Deduplication is algorithmic and must be checked for false merges and missed duplicates.
- Screening decisions are intentionally not inferred by the script.
- This API-based metadata search is a reproducible discovery aid, not a completed systematic review. It does not replace Web of Science/Scopus/IEEE Xplore/ACM DL/PubMed searches where appropriate, exportable database-specific query results, citation snowballing, full-text eligibility review, or independent reviewer screening.
- Do not state a PRISMA flow or novelty conclusion until searches are rerun and frozen, deduplication is checked, screening counts/reasons are completed, and full-text review is documented.

## Running

Run offline tests:

```bash
cd research/experiments
python -m unittest test_literature_search_export.py -v
```

Run live export:

```bash
python research/experiments/literature_search_export.py --out ./literature-search-export
```

Alternatively, use the GitHub Actions workflow-dispatch job after the workflow has merged; it uploads the exports as a retained artifact.
