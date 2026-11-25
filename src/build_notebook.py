"""Convert src/crop_yield_pipeline.py (cells marked with '# %%') into a Jupyter notebook.

Usage (from project root):
    python src/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace Crop_Yield_Prediction.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "crop_yield_pipeline.py").read_text(encoding="utf-8")

cells, buf, kind = [], [], "code"


def flush():
    text = "\n".join(buf).strip("\n")
    if not text:
        return
    if kind == "markdown":
        md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
        cells.append(nbf.v4.new_markdown_cell(md))
    else:
        cells.append(nbf.v4.new_code_cell(text))


for line in src.splitlines():
    if line.startswith("# %%"):
        flush()
        buf = []
        if "[markdown]" in line:
            kind = "markdown"
        else:
            kind = "code"
            title = line[4:].strip(" -")
            if title:
                cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
        continue
    buf.append(line)
flush()

# The notebook runs from the project root, where __file__ is undefined
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "Crop_Yield_Prediction.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} with {len(cells)} cells")
