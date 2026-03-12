import os
import sys


ROOT = '/Users/shivamsingh/Desktop/ResearchPaper'
sys.path.insert(0, ROOT)

from create_kaggle_notebooks import CELL_SAVE, CELL_SETUP, NOTEBOOKS, training_cell  # noqa: E402


OUT_PATH = os.path.join(ROOT, 'kaggle', 'PASTE_NOW.txt')


def build_text():
    lines = []
    for idx, nb in enumerate(NOTEBOOKS, start=1):
        title = nb['title']
        train = training_cell(nb['seeds'], nb['ablation'], nb['scenario'], nb['prefix'])
        lines.append("══════════════════════════════════════")
        lines.append(f"NOTEBOOK {idx}: {title}")
        lines.append("══════════════════════════════════════")
        lines.append("kaggle.com → Code → New Notebook")
        lines.append("Settings → Accelerator: GPU T4 x2")
        lines.append("Settings → Internet: ON")
        lines.append("Add Data → smartmarl-codebase → Add")
        lines.append("Delete existing cell. Add 3 new cells:")
        lines.append("")
        lines.append("CELL 1:")
        lines.append(CELL_SETUP.rstrip())
        lines.append("")
        lines.append("CELL 2:")
        lines.append(train.rstrip())
        lines.append("")
        lines.append("CELL 3:")
        lines.append(CELL_SAVE.rstrip())
        lines.append("")
        lines.append("Save & Run All. Then do next notebook.")
        lines.append("══════════════════════════════════════")
        lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    text = build_text()
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Wrote manual paste file: {OUT_PATH}")


if __name__ == '__main__':
    main()

