from __future__ import annotations

from pathlib import Path

import pytest


def test_scan_repo_reports_skip_xfail_and_quarantine_markers(tmp_path):
    from trellis.testing.hygiene import scan_repo

    test_root = tmp_path / "tests" / "test_sample.py"
    test_root.parent.mkdir(parents=True, exist_ok=True)
    test_root.write_text(
        """
import pytest

@pytest.mark.skip(reason="waiting on fixture")
def test_skip_decorator():
    assert True

@pytest.mark.xfail(reason="QUA-999 regression still open")
def test_ticketed_xfail():
    assert False

@pytest.mark.legacy_compat
def test_legacy():
    assert True

def test_runtime_skip():
    pytest.skip("host-specific precondition")

def test_importorskip():
    pytest.importorskip("QuantLib")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    findings = scan_repo(
        tmp_path,
        age_days_fn=lambda path: 12,
    )

    assert [(finding.kind, finding.bucket) for finding in findings] == [
        ("skip", "quarantine"),
        ("xfail", "quarantine"),
        ("legacy_compat", "quarantine"),
        ("skip", "quarantine"),
        ("importorskip", "quarantine"),
    ]
    assert findings[1].ticket == "QUA-999"
    assert findings[3].reason == "host-specific precondition"
    assert findings[4].reason == "QuantLib"


def test_stale_unticketed_xfails_only_flags_ancient_entries(tmp_path):
    from trellis.testing.hygiene import scan_repo, stale_unticketed_xfails

    test_root = tmp_path / "tests" / "test_xfail.py"
    test_root.parent.mkdir(parents=True, exist_ok=True)
    test_root.write_text(
        """
import pytest

@pytest.mark.xfail(reason="missing ticket")
def test_old_xfail():
    assert False

@pytest.mark.xfail(reason="QUA-321 tracked follow-on")
def test_ticketed_xfail():
    assert False
""".strip()
        + "\n",
        encoding="utf-8",
    )

    findings = scan_repo(
        tmp_path,
        age_days_fn=lambda path: 120,
    )
    violations = stale_unticketed_xfails(findings)

    assert len(violations) == 1
    assert violations[0].kind == "xfail"
    assert violations[0].bucket == "ancient"
    assert violations[0].ticket == ""


def test_hygiene_main_formats_report_and_fails_on_ancient_unticketed_xfail(tmp_path, capsys):
    from trellis.testing import hygiene

    test_root = tmp_path / "tests" / "test_xfail.py"
    test_root.parent.mkdir(parents=True, exist_ok=True)
    test_root.write_text(
        """
import pytest

@pytest.mark.xfail(reason="old xfail without ticket")
def test_old_xfail():
    assert False
""".strip()
        + "\n",
        encoding="utf-8",
    )

    original_scan_repo = hygiene.scan_repo

    def fake_scan_repo(root, **kwargs):
        return original_scan_repo(root, age_days_fn=lambda path: 150, **kwargs)

    hygiene.scan_repo = fake_scan_repo
    try:
        exit_code = hygiene.main(
            [
                "--root",
                str(tmp_path),
                "--fail-on-ancient-unticketed-xfail",
            ]
        )
    finally:
        hygiene.scan_repo = original_scan_repo

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Stale test hygiene report" in output
    assert "Ancient xfails without linked ticket ids:" in output
    assert "tests/test_xfail.py:3" in output


def test_collection_guard_rejects_ancient_unticketed_xfails(monkeypatch):
    import tests.conftest as root_conftest
    from trellis.testing.hygiene import HygieneFinding

    finding = HygieneFinding(
        path=str(root_conftest.REPO_ROOT / "tests" / "test_sample.py"),
        line=17,
        kind="xfail",
        source="decorator",
        reason="temporary workaround",
        ticket="",
        age_days=150,
        bucket="ancient",
        requires_ticket=True,
    )
    monkeypatch.setattr(root_conftest, "scan_repo", lambda *args, **kwargs: [finding])

    with pytest.raises(pytest.UsageError, match="Ancient xfails without linked ticket ids are not allowed"):
        root_conftest.pytest_collection_modifyitems(None, [])


def test_collection_guard_allows_override_for_local_debugging(monkeypatch):
    import tests.conftest as root_conftest
    from trellis.testing.hygiene import HygieneFinding

    finding = HygieneFinding(
        path=str(root_conftest.REPO_ROOT / "tests" / "test_sample.py"),
        line=17,
        kind="xfail",
        source="decorator",
        reason="temporary workaround",
        ticket="",
        age_days=150,
        bucket="ancient",
        requires_ticket=True,
    )
    monkeypatch.setattr(root_conftest, "scan_repo", lambda *args, **kwargs: [finding])
    monkeypatch.setenv("TRELLIS_ALLOW_STALE_XFAIL", "1")

    root_conftest.pytest_collection_modifyitems(None, [])
