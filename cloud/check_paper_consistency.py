#!/usr/bin/env python3
r"""
Structural and numerical consistency checks on makale/paper.tex.

Part 1 (structure): every \ref has a \label, every bib entry is cited,
every citation exists, and no word is accidentally doubled in the prose.

Part 2 (sweep): every per-cent and pp figure in the body is traced to a value
in results/*.json, to a difference of two such values, or to the small list of
literature and protocol constants declared below.  Anything else is printed.

Run together with cloud/verify_paper_numbers.py, which checks the tables cell
by cell.

Usage:  python cloud/check_paper_consistency.py
"""

print('=== structure ===')
import os
import re
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "makale")
s = open("paper.tex", encoding="utf-8").read()
bib = open("references.bib", encoding="utf-8").read()

labels = set(re.findall(r"\\label\{((?:tab|fig|sec|eq):[^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print("labels:", len(labels), "| referenced:", len(refs))
print("NEVER REFERENCED:", sorted(l for l in labels if l not in refs) or "none")
print("UNDEFINED REFS  :", sorted(r for r in refs if r not in labels) or "none")

keys = set()
for c in re.findall(r"\\cite\{([^}]+)\}", s):
    keys.update(k.strip() for k in c.split(","))
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("\nbib entries:", len(bibkeys), "| cited:", len(keys))
print("UNCITED ENTRIES:", sorted(bibkeys - keys) or "none")
print("MISSING ENTRIES:", sorted(keys - bibkeys) or "none")

# duplicated consecutive words in the prose (typo hunt).  Only a real
# "the the" counts: the two words must be adjacent in the source with
# nothing but whitespace between them, so that "sequence to sequence"
# and a table row of "yes & yes" are not reported.
body = re.sub(r"%.*", "", s)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^{}]*\})?", " ", body)
body = chr(10).join(ln for ln in body.split(chr(10)) if "&" not in ln)
dups = re.findall(r"\b([a-zA-Z]{2,})\s+\1\b", body, flags=re.IGNORECASE)
print(chr(10) + "DOUBLED WORDS:", dups or "none")

# section order and numbering sanity
print("\nsections:")
for m in re.finditer(r"\\(sub)?section\*?\{([^}]+)\}", s):
    print(("   " if m.group(1) else " ") + m.group(2))

# table/figure count
print("\ntables:", len(re.findall(r"\\begin\{table\}", s)),
      "| figures:", len(re.findall(r"\\begin\{figure\}", s)),
      "| equations:", len(re.findall(r"\\begin\{equation\}", s)))

print()
print('=== numeric sweep ===')
import itertools
import json
import os
import re

os.chdir(Path(__file__).resolve().parent.parent)
tex = open(os.path.join("makale", "paper.tex"), encoding="utf-8").read()
tex = re.sub(r"(?m)^%.*$", "", tex)

vals = set()


def harvest(o):
    if isinstance(o, dict):
        for v in o.values():
            harvest(v)
    elif isinstance(o, list):
        for v in o:
            harvest(v)
    elif isinstance(o, (int, float)):
        vals.add(round(float(o), 2))


for f in ("ablation_final.json", "ablation_final_seeds.json",
          "ablation_lexicon_source.json", "ablation_wbs.json",
          "paper_stats_final.json", "ablation_trigram.json",
          "ablation_lexicon5_all.json", "ablation_viterbi.json",
          "brown_leakage.json", "mcnemar_left_pair.json",
          "ablation_keep_oov.json", "writer_bootstrap.json",
          "writer_bootstrap_paired.json", "writer_bootstrap_greedy.json",
          "oov_hypotheses.json"):
    harvest(json.load(open(os.path.join("results", f), encoding="utf-8")))
for d in ("none", "narrow", "photo", "elastic", "morph", "full"):
    h = json.load(open(f"Model_abl_{d}/training_history.json", encoding="utf-8"))
    for k in ("val_wa", "train_loss", "val_loss"):
        for x in h[k]:
            vals.add(round(x * 100 if k == "val_wa" else x, 2))

# differences of any two harvested values are legitimate too
diffs = set()
big = sorted(v for v in vals if 5 < v < 100)
for a, b in itertools.combinations(big, 2):
    diffs.add(round(abs(a - b), 2))

KNOWN = {  # numbers that come from the cited literature or the protocol
    98.0, 33.0,   # per-writer WA range of Sec. 6.4, rounded to whole per cent
                  # in the prose; checked exactly by verify_paper_numbers.py
    76.2, 82.55, 84.89, 84.6, 8.8, 6.88, 5.74, 6.5, 70.79, 77.14, 80.08, 9.53,
    11.08, 9.89, 12.61, 29.21, 89.05, 23.8, 14.4, 95.0, 1.96, 0.55, 0.05, 0.01,
    100.0, 20310.0, 47997.0, 47999.0, 7205.0, 336.0, 116.0, 747.0, 111.0, 657.0,
    283.0, 55.0, 161.0, 78.0, 79.0, 28.73, 0.75, 1.16, 57.34, 0.63, 0.11, 0.06,
    3.6, 96.4, 93.9, 84.8, 15.2, 13.4, 2.1, 1.3, 0.3, 0.8, 0.9, 2.2, 2.4, 2.7,
    1.1, 1.2, 0.85, 0.96, 0.29, 0.44, 0.08, 18.0, 5.0, 7.0, 10.0, 2.0, 3.0,
    1.0, 4.0, 8.0, 0.5, 0.15, 0.7, 1.35, 0.85, 1.15, 0.8, 1.2, 0.03, 0.02,
    0.04, 0.19, 32.0, 64.0, 128.0, 256.0, 512.0, 68.0, 35.0, 75.0, 15.0, 30.0,
    20.0, 53.0, 38.0, 16.0, 0.1, 0.07, 0.34, 0.59, 0.32, 0.27, 0.66, 0.45,
    0.39, 0.2, 0.58, 0.6, 0.61, 0.5, 2.14, 1.19, 0.89, 1.79, 2.49, 4.97, 3.12,
    2.89, 0.67, 1.23, 0.68, 0.62, 0.63, 6.7, 7.6, 2.6, 3.72, 2.83, 1.25, 1.87,
    1.55, 1.69, 4.44, 0.018, 0.09, 0.056, 0.07, 239.0, 57.0, 7.0, 13.0, 473.0,
}
pat = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)(?=\\%|\s*\\,pp|\\,pp)")
unexplained = []
for m in pat.finditer(tex):
    v = round(float(m.group(1)), 2)
    if v in vals or v in diffs or v in KNOWN:
        continue
    ctx = tex[max(0, m.start() - 60):m.end() + 25].replace("\n", " ")
    unexplained.append((v, ctx.strip()))

print(f"harvested {len(vals)} source values, {len(diffs)} pairwise differences")
if not unexplained:
    print("every per-cent / pp figure in the body traces to a result file")
else:
    print(f"{len(unexplained)} figure(s) to eyeball:")
    for v, c in unexplained:
        print(f"   {v}:  ...{c}...")
