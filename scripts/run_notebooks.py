"""Выполнить notebooks с сохранением реальных outputs и статуса выполнения."""
import argparse
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all', action='store_true', help='Включить BERT notebooks 8–10; нужны pretrained-веса и результаты обучения')
    args = p.parse_args()
    rows = []
    for path in sorted((ROOT / 'notebooks').glob('week_*.ipynb')):
        week = int(path.stem.split('_')[1])
        if not args.all and week in [8, 9, 10]:
            rows.append({'notebook': path.name, 'status': 'not_run', 'reason': 'Requires pretrained BERT and completed experiments'})
            continue
        print('Executing', path.name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        try:
            NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
        except Exception:
            nbformat.write(notebook, path)
            rows.append({'notebook': path.name, 'status': 'failed'})
            (ROOT / 'reports/notebook_execution.json').write_text(json.dumps(rows, indent=2))
            raise
        nbformat.write(notebook, path)
        rows.append({'notebook': path.name, 'status': 'passed'})
    (ROOT / 'reports/notebook_execution.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
