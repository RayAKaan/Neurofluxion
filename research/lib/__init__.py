"""Shared research-infrastructure library for Neurofluxion evidence generation.

This package contains research-only tooling: provenance capture, evidence
artifact writing, descriptive statistics, numerical comparison helpers and a
canonical registry of simulator graph cases.

It never modifies runtime/application source. Experiment scripts import from
here so that every result carries the same provenance and schema contract.
"""
from __future__ import annotations

SCHEMA_NAMESPACE = "neurofluxion.research"

__all__ = ["SCHEMA_NAMESPACE"]