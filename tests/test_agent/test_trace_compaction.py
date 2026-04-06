"""Tests for cold-tier trace compaction and gap aggregation (QUA-405)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import yaml
import pytest


def _write_trace(traces_dir: Path, instrument: str, method: str, ts: datetime, resolved: bool = False, lesson_id: str | None = None):
    """Write a minimal trace file matching the naming convention."""
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    filename = f"{ts_str}_{instrument}_{method}.yaml"
    data = {
        "timestamp": ts.isoformat(),
        "instrument": instrument,
        "method": method,
        "resolved": resolved,
        "lesson_id": lesson_id,
    }
    path = traces_dir / filename
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


class TestCompactTraces:
    """Verify compact_traces groups, summarises, and removes old trace files."""

    def test_compact_old_traces(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_SUMMARIES_DIR", traces_dir / "summaries")

        now = datetime.now()
        old = now - timedelta(days=45)

        _write_trace(traces_dir, "bond", "analytical", old, resolved=True, lesson_id="fd_001")
        _write_trace(traces_dir, "bond", "analytical", old - timedelta(hours=1), resolved=False)
        _write_trace(traces_dir, "swaption", "monte_carlo", old, resolved=True)
        # Recent trace should be kept
        _write_trace(traces_dir, "bond", "analytical", now - timedelta(days=5), resolved=True)

        stats = promotion.compact_traces(older_than_days=30)

        assert stats["cohorts_summarized"] == 2
        assert stats["traces_compacted"] == 3
        assert stats["traces_kept"] == 1

        summary_path = traces_dir / "summaries" / "bond_analytical.yaml"
        assert summary_path.exists()
        summary = yaml.safe_load(summary_path.read_text())
        assert summary["total_attempts"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert "fd_001" in summary["lesson_ids"]

    def test_compact_empty_dir(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_SUMMARIES_DIR", traces_dir / "summaries")

        stats = promotion.compact_traces(older_than_days=30)
        assert stats == {"cohorts_summarized": 0, "traces_compacted": 0, "traces_kept": 0}

    def test_compact_nonexistent_dir(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        monkeypatch.setattr(promotion, "_TRACES_DIR", tmp_path / "missing")
        stats = promotion.compact_traces()
        assert stats == {"cohorts_summarized": 0, "traces_compacted": 0, "traces_kept": 0}

    def test_compact_incremental(self, tmp_path, monkeypatch):
        """Running compact_traces twice merges summaries correctly."""
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_SUMMARIES_DIR", traces_dir / "summaries")

        now = datetime.now()
        old = now - timedelta(days=45)

        _write_trace(traces_dir, "cap", "analytical", old, resolved=True)
        promotion.compact_traces(older_than_days=30)

        _write_trace(traces_dir, "cap", "analytical", old - timedelta(hours=2), resolved=False, lesson_id="fd_010")
        stats = promotion.compact_traces(older_than_days=30)

        summary = yaml.safe_load((traces_dir / "summaries" / "cap_analytical.yaml").read_text())
        assert summary["total_attempts"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1

    def test_compact_old_platform_event_logs_into_summary_yaml(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from trellis.agent.knowledge import promotion
        from trellis.agent.platform_traces import (
            append_platform_trace_event,
            load_platform_trace_events,
            load_platform_trace_payload,
            record_platform_trace,
        )

        traces_dir = tmp_path / "traces"
        platform_dir = traces_dir / "platform"
        platform_dir.mkdir(parents=True)
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_SUMMARIES_DIR", traces_dir / "summaries")

        compiled = SimpleNamespace(
            request=SimpleNamespace(
                request_id="executor_build_compaction_demo",
                request_type="build",
                entry_point="executor",
                instrument_type="european_option",
                metadata={},
            ),
            execution_plan=SimpleNamespace(
                action="build_then_price",
                route_method="analytical",
                measures=(),
                requires_build=True,
            ),
            pricing_plan=SimpleNamespace(sensitivity_support=None),
            product_ir=SimpleNamespace(instrument="european_option"),
            blocker_report=None,
            knowledge_summary={},
        )

        trace_path = record_platform_trace(
            compiled,
            success=True,
            outcome="build_completed",
            root=platform_dir,
        )
        append_platform_trace_event(
            compiled,
            "review_completed",
            status="ok",
            details={"note": "kept for regression"},
            root=platform_dir,
        )

        events_path = trace_path.with_suffix(".events.ndjson")
        old_time = (datetime.now() - timedelta(days=45)).timestamp()
        import os

        os.utime(trace_path, (old_time, old_time))
        os.utime(events_path, (old_time, old_time))

        stats = promotion.compact_traces(older_than_days=30)
        payload = load_platform_trace_payload(trace_path)
        events = load_platform_trace_events(trace_path)

        assert stats["cohorts_summarized"] == 0
        assert stats["traces_compacted"] == 1
        assert stats["traces_kept"] == 0
        assert not events_path.exists()
        assert [event["event"] for event in payload["events"]] == [
            "request_succeeded",
            "review_completed",
        ]
        assert [event.event for event in events] == [
            "request_succeeded",
            "review_completed",
        ]


class TestGapAggregation:
    """Verify _record_gap_aggregated deduplicates and increments."""

    def test_new_gap(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_GAP_REGISTRY_PATH", traces_dir / "gap_registry.yaml")

        gap_id = promotion._record_gap_aggregated(
            "Missing local vol calibrator",
            method="monte_carlo",
            features=["local_vol", "calibration"],
        )
        assert gap_id.startswith("gap_")

        registry = yaml.safe_load((traces_dir / "gap_registry.yaml").read_text())
        assert len(registry) == 1
        assert registry[0]["occurrences"] == 1
        assert registry[0]["status"] == "open"

    def test_deduplication(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_GAP_REGISTRY_PATH", traces_dir / "gap_registry.yaml")

        gap_id_1 = promotion._record_gap_aggregated(
            "Missing local vol calibrator",
            method="monte_carlo",
            features=["local_vol"],
        )
        gap_id_2 = promotion._record_gap_aggregated(
            "Missing local vol calibrator",
            method="monte_carlo",
            features=["local_vol"],
        )
        assert gap_id_1 == gap_id_2

        registry = yaml.safe_load((traces_dir / "gap_registry.yaml").read_text())
        assert len(registry) == 1
        assert registry[0]["occurrences"] == 2

    def test_distinct_gaps(self, tmp_path, monkeypatch):
        from trellis.agent.knowledge import promotion

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        monkeypatch.setattr(promotion, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion, "_GAP_REGISTRY_PATH", traces_dir / "gap_registry.yaml")

        promotion._record_gap_aggregated("Gap A", method="analytical", features=[])
        promotion._record_gap_aggregated("Gap B", method="monte_carlo", features=[])

        registry = yaml.safe_load((traces_dir / "gap_registry.yaml").read_text())
        assert len(registry) == 2
