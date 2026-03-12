from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
PYTHON = str(ROOT / '.venv' / 'bin' / 'python')


class TestAutomation:
    def test_health_check_runs(self):
        result = subprocess.run(
            [PYTHON, 'monitor/health_check.py'],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert 'Health Report' in result.stdout

    def test_harvest_results_runs(self):
        result = subprocess.run(
            [PYTHON, 'kaggle/harvest_results.py', '--dry-run'],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0

    def test_auto_restart_script_exists(self):
        assert (ROOT / 'kaggle' / 'auto_restart.py').exists()
        assert (ROOT / 'kaggle' / 'auto_restart.sh').exists()

    def test_health_check_sh_is_executable(self):
        path = ROOT / 'health_check_now.sh'
        assert path.exists()
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR

    def test_finalize_results_imports_cleanly(self):
        result = subprocess.run(
            [PYTHON, '-c', 'import finalize_results'],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_daily_summary_creates_file(self):
        result = subprocess.run(
            [PYTHON, 'monitor/daily_summary.py'],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        today = ROOT / 'monitor' / 'today.txt'
        assert today.exists()
        content = today.read_text(encoding='utf-8')
        assert 'WHAT YOU NEED TO DO' in content

