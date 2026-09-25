#!/usr/bin/env python3
"""
How much of the reported precision survives if writers, not words, are the
sampling unit?

The Wilson intervals in the paper treat the 20,310 test words as independent
draws.  They are not: the words come from 161 writers, and a writer whose hand
the recognizer finds hard contributes a block of correlated errors.  This
script quantifies the difference with three bootstraps of the same per-word
outcomes -- resampling words, forms and writers -- and prints the Wilson
interval next to them.

The word-level bootstrap is the control: it must reproduce the Wilson interval,
which shows that the formula is doing exactly what its assumption implies and
that any gap to the writer-level interval comes from clustering, not from the
arithmetic.

Output: results/writer_bootstrap.json
Usage:  python cloud/writer_bootstrap.py [--preds results/preds_final/preds_full.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default="results/preds_final/preds_full.csv")
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/writer_bootstrap.json")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    c = z * z / (2 * n)
    r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return 100 * (p + c - r) / d, 100 * (p + c + r) / d


def main() -> int:
    a = parse_args()
    rng = np.random.default_rng(a.seed)

    fw = {}
    for ln in (ROOT / "aachen_splits" / "form_writer.txt").read_text(
            encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and len(ln.split()) >= 2:
            fw[ln.split()[0]] = ln.split()[1]

    with open(ROOT / a.preds, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_writer, by_form, flat = defaultdict(list), defaultdict(list), []
    for r in rows:
        form = "-".join(r["word_id"].split("-")[:2])
        c = int(r["correct"])
        flat.append(c)
        by_form[form].append(c)
        by_writer[fw[form]].append(c)
    flat = np.asarray(flat)
    n = len(flat)

    def boot(groups) -> np.ndarray:
        arrs = [np.asarray(v) for v in groups.values()]
        out = np.empty(a.reps)
        for b in range(a.reps):
            idx = rng.integers(0, len(arrs), len(arrs))
            out[b] = np.concatenate([arrs[i] for i in idx]).mean()
        return out

    def ci(samples) -> tuple:
        lo, hi = np.percentile(samples, [2.5, 97.5]) * 100
        return round(float(lo), 4), round(float(hi), 4), round(float(hi - lo) / 2, 4)

    res = {
        "preds": a.preds, "n_samples": n, "reps": a.reps, "seed": a.seed,
        "wa_pct": round(100 * float(flat.mean()), 4),
        "writers": len(by_writer), "forms": len(by_form),
    }
    wlo, whi = wilson(int(flat.sum()), n)
    res["wilson"] = {"lo": round(wlo, 4), "hi": round(whi, 4),
                     "half_width": round((whi - wlo) / 2, 4)}
    word = flat[rng.integers(0, n, (a.reps, n))].mean(axis=1)
    for name, s in (("word", word), ("form", boot(by_form)), ("writer", boot(by_writer))):
        lo, hi, hw = ci(s)
        res[f"bootstrap_{name}"] = {"lo": lo, "hi": hi, "half_width": hw}

    per_writer = np.asarray([100 * np.mean(v) for v in by_writer.values()])
    res["per_writer_wa_pct"] = {
        "min": round(float(per_writer.min()), 2),
        "median": round(float(np.median(per_writer)), 2),
        "max": round(float(per_writer.max()), 2),
        "sd": round(float(per_writer.std(ddof=1)), 2),
    }
    res["design_effect"] = round(
        res["bootstrap_writer"]["half_width"] / res["wilson"]["half_width"], 2)

    print(f"WA {res['wa_pct']:.2f}%   N={n}   {res['writers']} writers, "
          f"{res['forms']} forms")
    print(f"  Wilson                     "
          f"[{res['wilson']['lo']:.2f}, {res['wilson']['hi']:.2f}]  "
          f"+-{res['wilson']['half_width']:.2f} pp")
    for name in ("word", "form", "writer"):
        e = res[f"bootstrap_{name}"]
        print(f"  bootstrap over {name:<8s}    [{e['lo']:.2f}, {e['hi']:.2f}]  "
              f"+-{e['half_width']:.2f} pp")
    pw = res["per_writer_wa_pct"]
    print(f"  per-writer WA: min {pw['min']}  median {pw['median']}  max {pw['max']}"
          f"  (SD {pw['sd']})")
    print(f"  writer-level interval is {res['design_effect']}x the Wilson one")

    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dst, "w"), indent=2)
    print(f"-> {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
