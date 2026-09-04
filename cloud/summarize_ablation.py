#!/usr/bin/env python3
"""Ablation sonuclarini tablo halinde yazdirir."""
import csv, math, os, sys

LBL = {"narrow": "CRNN-L (baseline)", "photo": "+ wide photometric",
       "elastic": "+ elastic", "morph": "+ morphological",
       "full": "AugCRNN-T (proposed)"}
ORDER = ["narrow", "photo", "elastic", "morph", "full"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - e) * 100, (c + e) * 100


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"{'Configuration':<24}{'WA (%)':>9}{'95% CI':>20}{'k/n':>16}")
    print("-" * 70)
    seen = False
    for m in ORDER:
        p = os.path.join(root, f"Model_abl_{m}", "test_results_analysis.csv")
        if not os.path.exists(p):
            print(f"{LBL[m]:<24}{'-':>9}{'(kosulmadi)':>20}")
            continue
        seen = True
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        key = "correct" if "correct" in rows[0] else "Is_Correct"
        k = sum(1 for r in rows if str(r[key]).strip().lower() in ("1", "true"))
        n = len(rows)
        lo, hi = wilson(k, n)
        print(f"{LBL[m]:<24}{k/n*100:>9.2f}{f'[{lo:.2f}, {hi:.2f}]':>20}{f'{k}/{n}':>16}")
    if not seen:
        print("\n(henuz hicbir egitim tamamlanmamis)")
        return 1
    lex = os.path.join(root, "results", "ablation_lexicon.json")
    if os.path.exists(lex):
        import json
        r = json.load(open(lex))
        print(f"\n{'Lexicon / trigram':<44}{'WA (%)':>9}{'CER (%)':>9}")
        print("-" * 70)
        for c in r["configurations"]:
            print(f"{c['name']:<44}{c['wa_pct']:>9.2f}{c['cer_pct']:>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
