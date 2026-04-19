"""Extract tables and images from already-executed notebooks.

Reads notebook JSON without re-running, pulls out text/html tables and
image/png outputs, and writes them to a destination folder.
"""
import json
import os
import sys
import base64
import re
from pathlib import Path


def html_table_to_md(html: str) -> str:
    """Best-effort conversion of a pandas-style HTML table to GitHub MD."""
    try:
        import pandas as pd
        from io import StringIO
        tables = pd.read_html(StringIO(html))
        if not tables:
            return html
        return "\n\n".join(t.to_markdown(index=True) for t in tables)
    except Exception:
        return html


def extract(nb_path: str, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    nb = json.load(open(nb_path))
    tables_md = []
    image_paths = []
    for ci, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        for oi, out in enumerate(cell.get('outputs', [])):
            data = out.get('data', {})
            # text/html tables
            if 'text/html' in data:
                html = data['text/html']
                if isinstance(html, list):
                    html = ''.join(html)
                if '<table' in html:
                    md = html_table_to_md(html)
                    tables_md.append((ci, oi, md))
            # image/png
            if 'image/png' in data:
                b64 = data['image/png']
                if isinstance(b64, list):
                    b64 = ''.join(b64)
                fname = f"{tag}_c{ci:02d}_o{oi:02d}.png"
                fp = os.path.join(out_dir, fname)
                with open(fp, 'wb') as f:
                    f.write(base64.b64decode(b64))
                image_paths.append(fname)
    # write table file
    if tables_md:
        md_path = os.path.join(out_dir, f"{tag}_tables.md")
        with open(md_path, 'w') as f:
            f.write(f"# Tables extracted from `{os.path.basename(nb_path)}`\n\n")
            for ci, oi, md in tables_md:
                f.write(f"## Cell {ci}, output {oi}\n\n")
                f.write(md)
                f.write("\n\n")
    return len(tables_md), len(image_paths)


if __name__ == '__main__':
    nb_path = sys.argv[1]
    out_dir = sys.argv[2]
    tag = sys.argv[3]
    nt, ni = extract(nb_path, out_dir, tag)
    print(f"{nb_path}: {nt} tables, {ni} images -> {out_dir}")
