"""
Generate notebooks/01_preprocessing_metrics.ipynb, 02_classifier_metrics.ipynb, 03_model_comparison.ipynb,
04_finetuning.ipynb.

The notebooks are thin: all logic lives in src/evaluation/ (data, metrics, baselines, tgat_boostnet,
stats, plotting, preprocessing, literature). Edit the cell sources in notebooks/_cells/*.py
(cells separated by '# %%' and '# %% [markdown]' markers) and re-run:

    .venv\\Scripts\\python.exe scripts\\build_notebooks.py
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CELL_DIR = os.path.join(ROOT, "notebooks", "_cells")
NAMES = ["01_preprocessing_metrics", "02_classifier_metrics", "03_model_comparison", "04_finetuning"]


def parse(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    shared = os.path.join(CELL_DIR, "_setup.py")
    with open(shared, encoding="utf-8") as f:
        text = text.replace("# %% INCLUDE_SETUP", f.read())
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if header.strip().startswith("[markdown]"):
            src = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body})
    return cells


def main():
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for n in NAMES:
        nb = {"cells": parse(os.path.join(CELL_DIR, f"{n}.py")),
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}, "accelerator": "GPU",
                           "colab": {"provenance": [], "gpuType": "T4"}},
              "nbformat": 4, "nbformat_minor": 5}
        for c in nb["cells"]:
            c["source"] = c["source"].splitlines(keepends=True)
        with open(os.path.join(ROOT, "notebooks", f"{n}.ipynb"), "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("wrote", f"notebooks/{n}.ipynb", len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
