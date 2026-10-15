"""Check that every number in the README/report tables comes from results/.

A table cell token like 0.9426, 0.185 or 138,736 passes if some value in
results/*.csv or results/*.json rounds to it at the printed precision. A table
directly preceded by `<!-- check_numbers: skip -->` is not checked (used for the
historical Colab table, whose numbers live in that notebook's outputs).

Usage: python tools/check_numbers.py README.md report.md   (exit 1 on any miss)
"""
import glob
import json
import re
import sys

import pandas as pd

TOKEN = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+)(?![\w.])")
SKIP = "<!-- check_numbers: skip -->"


def result_values():
    vals = []
    for path in glob.glob("results/*.csv"):
        df = pd.read_csv(path)
        vals += [float(v) for v in df.select_dtypes("number").to_numpy().ravel() if v == v]
    for path in glob.glob("results/*.json"):
        vals += [float(v) for v in json.load(open(path)).values() if isinstance(v, (int, float))]
    return vals


def table_tokens(text):
    lines = text.splitlines()
    out, skip, prev = [], False, ""
    for i, line in enumerate(lines):
        if line.startswith("|"):
            if not prev.startswith("|"):
                k = i - 1
                while k >= 0 and not lines[k].strip():
                    k -= 1
                skip = k >= 0 and lines[k].strip() == SKIP
            if not skip and not re.match(r"^\|[\s:|-]+\|$", line):
                out += [(i + 1, t) for t in TOKEN.findall(line)]
        prev = line
    return out


def matches(token, vals):
    t = token.replace("−", "-").replace(",", "").lstrip("+")
    v = float(t)
    d = len(t.split(".")[1]) if "." in t else 0
    return any(round(x, d) == v for x in vals)


def main(paths):
    vals = result_values()
    misses = []
    for path in paths:
        for line, tok in table_tokens(open(path).read()):
            if not matches(tok, vals):
                misses.append(f"{path}:{line}: {tok}")
    print("\n".join(misses) or f"all table numbers in {', '.join(paths)} trace to results/")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
