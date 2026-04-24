from __future__ import annotations

from pathlib import Path


def test_build_task_batch_report_normalizes_paths_and_failed_rows():
    from trellis.agent.task_batch_reports import build_task_batch_report

    root = Path(__file__).resolve().parents[2]
    results = [
        {
            "task_id": "F001",
            "title": "FinancePy parity",
            "task_corpus": "benchmark_financepy",
            "success": True,
            "attempts": 0,
            "elapsed_seconds": 12.3,
            "token_usage_summary": {"total_tokens": 42},
            "task_run_history_path": str(root / "task_runs" / "history" / "F001" / "run.json"),
        },
        {
            "task_id": "P004",
            "title": "Callable collar",
            "task_corpus": "extension",
            "success": False,
            "attempts": 2,
            "elapsed_seconds": 34.5,
            "token_usage_summary": {"total_tokens": 84},
            "task_diagnosis_failure_bucket": "blocked",
            "task_diagnosis_headline": "Callable collar failed in blocked.",
            "task_diagnosis_next_action": "Inspect the blocking primitive.",
            "task_diagnosis_packet_path": str(
                root / "task_runs" / "diagnostics" / "history" / "P004" / "run.json"
            ),
        },
    ]

    report = build_task_batch_report(
        results,
        collection_name="Manual batch",
        model="gpt-5.4-mini",
        validation="standard",
        force_rebuild=False,
        fresh_build=False,
        knowledge_light=False,
        selection={"selection_mode": "ids", "status": "all", "requested_task_ids": ("F001", "P004")},
        raw_results_path=root / "task_results_manual.json",
        summary_path=root / "task_results_manual_summary.json",
        root=root,
    )

    assert report["summary"]["totals"]["tasks"] == 2
    assert report["artifacts"]["raw_results_path"] == "task_results_manual.json"
    assert report["tasks"][0]["history_path"] == "task_runs/history/F001/run.json"
    assert report["failed_tasks"][0]["diagnosis_packet_path"] == "task_runs/diagnostics/history/P004/run.json"
    assert report["failed_tasks"][0]["status_bucket"] == "blocked"


def test_render_task_batch_markdown_includes_selection_and_gap_signals():
    from trellis.agent.task_batch_reports import build_task_batch_report, render_task_batch_markdown

    report = build_task_batch_report(
        [
            {
                "task_id": "P004",
                "title": "Callable collar",
                "task_corpus": "extension",
                "success": False,
                "attempts": 2,
                "elapsed_seconds": 34.5,
                "task_diagnosis_failure_bucket": "blocked",
                "task_diagnosis_headline": "Callable collar failed in blocked.",
                "task_diagnosis_next_action": "Inspect the blocking primitive.",
                "token_usage_summary": {"total_tokens": 84},
            }
        ],
        collection_name="Manual batch",
        model="gpt-5.4-mini",
        validation="standard",
        force_rebuild=False,
        fresh_build=False,
        knowledge_light=False,
        selection={
            "selection_mode": "ids",
            "status": "all",
            "corpora": ("extension",),
            "requested_task_ids": ("P004",),
        },
    )

    markdown = render_task_batch_markdown(report, max_task_rows=10)

    assert "# Manual batch" in markdown
    assert "- Mode: `ids`" in markdown
    assert "- Corpora: `extension`" in markdown
    assert "## Gap Signals" in markdown
    assert "`P004` `blocked`" in markdown
    assert "| `P004` | `extension` | `blocked` |" in markdown
