"""Tests for the generated unified skill layer."""

from __future__ import annotations

from dataclasses import asdict

from trellis.agent.knowledge.skills import (
    clear_skill_index_cache,
    get_skill_record,
    get_skill_lineage,
    load_skill_index,
    load_skill_lineage_index,
    query_skill_records,
)


def test_skill_index_manifest_and_kinds_are_present():
    index = load_skill_index()

    assert index.manifest.repo_revision
    assert index.manifest.record_count == len(index.records)
    assert index.manifest.kind_counts["lesson"] > 0
    assert index.manifest.kind_counts["principle"] > 0
    assert index.manifest.kind_counts["cookbook"] > 0
    assert index.manifest.kind_counts["route_hint"] > 0


def test_lesson_skill_projection_keeps_core_metadata():
    record = get_skill_record("lesson:mc_020")

    assert record is not None
    assert record.kind == "lesson"
    assert record.origin == "captured"
    assert record.source_artifact == "mc_020"
    assert "monte_carlo" in record.method_families
    assert "multi_asset" in record.concepts
    assert "severity:critical" in record.tags


def test_principle_skill_projection_inherits_lesson_context():
    record = get_skill_record("principle:P11")

    assert record is not None
    assert record.kind == "principle"
    assert record.origin == "derived"
    assert record.lineage_status == "derived"
    assert "principles.derived_from" in record.lineage_evidence
    assert "lesson:sem_001" in record.parents
    assert "lesson:mc_020" in record.parents
    assert record.method_families
    assert record.concepts


def test_cookbook_skill_projection_aggregates_contract_metadata():
    record = get_skill_record("cookbook:analytical")

    assert record is not None
    assert record.kind == "cookbook"
    assert record.lineage_status == "source_root"
    assert record.lineage_evidence == ("cookbooks.entry",)
    assert "cds" in record.instrument_types
    assert "credit_curve" in record.concepts
    assert "solution_contract:credit_default_swap" in record.tags


def test_route_hint_projection_includes_instruction_lifecycle_records():
    route_skills = query_skill_records(
        kind="route_hint",
        method_family="monte_carlo",
        instrument_type="cds",
        route_family="credit_default_swap",
    )
    skill_ids = {record.skill_id for record in route_skills}
    summaries = [record.summary for record in route_skills]

    assert "route_hint:credit_default_swap_monte_carlo:route-helper" in skill_ids
    assert "route_hint:credit_default_swap_monte_carlo:schedule-builder" in skill_ids
    assert "route_hint:credit_default_swap_monte_carlo:schedule-body" in skill_ids
    assert not any("default-time" in summary.lower() or "default time" in summary.lower() for summary in summaries)


def test_route_hint_lineage_links_back_to_matching_cookbook():
    record = get_skill_record("route_hint:analytical_garman_kohlhagen:note:1")

    assert record is not None
    assert record.lineage_status == "derived"
    assert "route.match_method_to_cookbook" in record.lineage_evidence
    assert "cookbook:analytical" in record.parents


def test_skill_lineage_query_surfaces_children_and_same_source_records():
    lineage = get_skill_lineage("cookbook:analytical")

    assert lineage is not None
    assert "route_hint:analytical_garman_kohlhagen:note:1" in lineage["children"]

    route_lineage = get_skill_lineage("route_hint:analytical_garman_kohlhagen:note:1")
    assert route_lineage is not None
    assert "route_hint:analytical_garman_kohlhagen:route-helper" in route_lineage["same_source"]

    lineage_index = load_skill_lineage_index()
    assert lineage_index["cookbook:analytical"]["children"] == lineage["children"]


def test_ambiguous_route_hint_cookbook_backfill_is_left_unresolved():
    record = get_skill_record("route_hint:nth_to_default_monte_carlo:schedule-builder")

    assert record is not None
    assert record.parents == ()
    assert record.lineage_status == "advisory"
    assert record.lineage_evidence == ("route.match_method_to_cookbook_ambiguous",)


def test_skill_index_generation_is_deterministic():
    clear_skill_index_cache()
    first = asdict(load_skill_index())
    clear_skill_index_cache()
    second = asdict(load_skill_index())

    assert first == second
