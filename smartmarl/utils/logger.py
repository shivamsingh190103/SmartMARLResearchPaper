"""Minimal experiment logger for SmartMARL."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict


class ExperimentLogger:
    def __init__(self, log_dir: str = "results", name: str = "smartmarl") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def log_row(self, filename: str, row: Dict) -> None:
        path = self.log_dir / filename
        write_header = not path.exists()

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
