"""Tests for ExperimentLogger."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from smartmarl.utils.logger import ExperimentLogger


class TestExperimentLogger:
    def test_creates_log_dir(self, tmp_path):
        log_dir = str(tmp_path / "new_logs")
        logger = ExperimentLogger(log_dir=log_dir, name="test_create_dir")
        assert Path(log_dir).is_dir()

    def test_info_does_not_raise(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_info")
        logger.info("Hello world")  # Should not raise

    def test_log_row_creates_file(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_log_row")
        logger.log_row("results.csv", {"episode": 1, "reward": 42.0})
        assert (tmp_path / "results.csv").exists()

    def test_log_row_writes_header(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_header")
        logger.log_row("data.csv", {"a": 1, "b": 2})
        with open(tmp_path / "data.csv", newline="") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {"a", "b"}

    def test_log_row_writes_values(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_values")
        logger.log_row("data.csv", {"x": 10, "y": 20})
        with open(tmp_path / "data.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["x"] == "10"
        assert rows[0]["y"] == "20"

    def test_log_row_appends_without_duplicate_header(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_append")
        logger.log_row("out.csv", {"ep": 1, "reward": 1.0})
        logger.log_row("out.csv", {"ep": 2, "reward": 2.0})
        with open(tmp_path / "out.csv", newline="") as f:
            lines = f.readlines()
        # Should be 1 header + 2 data rows = 3 lines
        assert len(lines) == 3

    def test_log_row_multiple_rows_correct_values(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), name="test_multi")
        rows_in = [{"step": i, "loss": float(i) * 0.1} for i in range(5)]
        for row in rows_in:
            logger.log_row("train.csv", row)
        with open(tmp_path / "train.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows_out = list(reader)
        assert len(rows_out) == 5
        for i, row in enumerate(rows_out):
            assert row["step"] == str(i)

    def test_nested_log_dir_created(self, tmp_path):
        deep_dir = str(tmp_path / "a" / "b" / "c")
        logger = ExperimentLogger(log_dir=deep_dir, name="test_nested")
        assert Path(deep_dir).is_dir()
