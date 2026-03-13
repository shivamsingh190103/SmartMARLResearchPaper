"""Generate manual Kaggle notebook creation instructions from local .ipynb files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.common import notebook_local_slugs


ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
OUT_PATH = ROOT / 'kaggle' / 'MANUAL_STEPS.txt'


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def _load_cell_sources(ipynb_path: Path) -> List[str]:
    data = json.loads(ipynb_path.read_text(encoding='utf-8'))
    cells = data.get('cells', [])
    sources: List[str] = []
    for cell in cells:
        src = cell.get('source', [])
        if isinstance(src, list):
            sources.append(''.join(src))
        elif isinstance(src, str):
            sources.append(src)
        else:
            sources.append('')
    return sources


def build_manual_text() -> str:
    notebooks = notebook_local_slugs()
    lines: List[str] = []
    lines.append('========================================')
    lines.append('MANUAL KAGGLE NOTEBOOK CREATION STEPS')
    lines.append('========================================')
    lines.append('')
    lines.append('If automatic push failed, create notebooks manually:')
    lines.append('')
    lines.append('1. Go to kaggle.com → Code → + New Notebook')
    lines.append('2. Settings → Accelerator → GPU T4 x2')
    lines.append('3. Settings → Internet → On')
    lines.append('4. Add Data → Search "smartmarl-codebase" → Add')
    lines.append('5. Paste the code below into the notebook cells')
    lines.append('6. Click "Save & Run All"')
    lines.append(f'7. Repeat for each notebook ({len(notebooks)} total)')
    if not notebooks:
        lines.append('   (No notebook assets found. Run: python kaggle/create_notebooks.py)')
    lines.append('')

    for idx, nb in enumerate(notebooks, start=1):
        ipynb = NOTEBOOK_ROOT / nb / 'notebook.ipynb'
        lines.append('----------------------------------------')
        lines.append(f'NOTEBOOK {idx}: {nb}')
        lines.append('----------------------------------------')
        lines.append('')

        if not ipynb.exists():
            lines.append(f'MISSING FILE: {ipynb}')
            lines.append('')
            continue

        cells = _load_cell_sources(ipynb)
        labels = ['SETUP', 'TRAINING', 'SAVE OUTPUTS']
        for cidx, src in enumerate(cells[:3], start=1):
            label = labels[cidx - 1] if cidx - 1 < len(labels) else f'CELL {cidx}'
            lines.append(f'--- CELL {cidx}: {label} ---')
            lines.append(src.rstrip())
            lines.append('')

    lines.append('========================================')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    text = build_manual_text()
    _atomic_write(OUT_PATH, text)
    print(f'Wrote manual instructions: {OUT_PATH}')


if __name__ == '__main__':
    main()
