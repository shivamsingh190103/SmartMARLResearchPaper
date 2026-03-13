from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor.common import notebook_local_slugs  # noqa: E402

NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
OUT_PATH = ROOT / 'kaggle' / 'PASTE_NOW.txt'


def _load_cells(ipynb: Path) -> List[str]:
    data = json.loads(ipynb.read_text(encoding='utf-8'))
    out: List[str] = []
    for cell in data.get('cells', []):
        src = cell.get('source', [])
        if isinstance(src, list):
            out.append(''.join(src))
        elif isinstance(src, str):
            out.append(src)
        else:
            out.append('')
    return out


def build_text() -> str:
    lines: List[str] = []
    slugs = notebook_local_slugs()
    if not slugs:
        return (
            'No generated notebook assets found.\n'
            'Run: python kaggle/create_notebooks.py\n'
        )

    for idx, slug in enumerate(slugs, start=1):
        ipynb = NOTEBOOK_ROOT / slug / 'notebook.ipynb'
        lines.append('======================================')
        lines.append(f'NOTEBOOK {idx}: {slug}')
        lines.append('======================================')
        lines.append('kaggle.com -> Code -> New Notebook')
        lines.append('Settings -> Accelerator: GPU T4 x2')
        lines.append('Settings -> Internet: ON')
        lines.append('Add Data -> smartmarl-codebase -> Add')
        lines.append('Delete existing cell. Add 3 new code cells:')
        lines.append('')

        if not ipynb.exists():
            lines.append(f'MISSING: {ipynb}')
            lines.append('')
            continue

        cells = _load_cells(ipynb)
        for cidx, src in enumerate(cells[:3], start=1):
            lines.append(f'CELL {cidx}:')
            lines.append(src.rstrip())
            lines.append('')

        lines.append('Save & Run All. Then do next notebook.')
        lines.append('')

    return '\n'.join(lines)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_text(), encoding='utf-8')
    print(f'Wrote manual paste file: {OUT_PATH}')


if __name__ == '__main__':
    main()
