"""Contract tests for cassette-availability helpers."""

from __future__ import annotations

from tests.test_contracts import conftest as contract_conftest


class TestCassetteAvailabilityHelpers:
    def test_cassette_available_respects_record_cassette_false(self, monkeypatch, tmp_path):
        cassette_path = tmp_path / "T38.yaml"
        cassette_path.write_text("meta: {}\n", encoding="utf-8")

        monkeypatch.setitem(
            contract_conftest.CANARY_META,
            "T38",
            {"id": "T38", "record_cassette": False},
        )
        monkeypatch.setattr(contract_conftest, "cassette_path_for", lambda task_id: cassette_path)

        assert contract_conftest.cassette_available("T38") is False

    def test_cassette_available_falls_back_to_file_existence_without_canary_metadata(
        self, monkeypatch, tmp_path
    ):
        cassette_path = tmp_path / "T01.yaml"
        cassette_path.write_text("meta: {}\n", encoding="utf-8")

        monkeypatch.delitem(contract_conftest.CANARY_META, "T01", raising=False)
        monkeypatch.setattr(contract_conftest, "cassette_path_for", lambda task_id: cassette_path)

        assert contract_conftest.cassette_available("T01") is True

    def test_full_task_cassette_available_respects_record_cassette_false(
        self, monkeypatch, tmp_path
    ):
        cassette_path = tmp_path / "T38.yaml"
        cassette_path.write_text("meta: {}\n", encoding="utf-8")

        monkeypatch.setitem(
            contract_conftest.CANARY_META,
            "T38",
            {"id": "T38", "record_cassette": False},
        )
        monkeypatch.setattr(
            contract_conftest,
            "full_task_cassette_path_for",
            lambda task_id: cassette_path,
        )

        assert contract_conftest.full_task_cassette_available("T38") is False

    def test_full_task_cassette_available_falls_back_to_file_existence_without_metadata(
        self, monkeypatch, tmp_path
    ):
        cassette_path = tmp_path / "T13.yaml"
        cassette_path.write_text("meta: {}\n", encoding="utf-8")

        monkeypatch.delitem(contract_conftest.CANARY_META, "T13", raising=False)
        monkeypatch.setattr(
            contract_conftest,
            "full_task_cassette_path_for",
            lambda task_id: cassette_path,
        )

        assert contract_conftest.full_task_cassette_available("T13") is True
