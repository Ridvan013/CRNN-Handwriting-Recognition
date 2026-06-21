"""
Genisletilmis McNemar analizi - Aachen CSV (4 sutun) icin.
test_results_analysis.csv 4 decoder konfigurasyonunu logluyor:
  - Raw_Predicted_Text     = greedy only
  - Predicted_Text         = greedy + trigram (ANA PIPELINE)
  - Beam_Predicted_Text    = beam k=10 only
  - Beam_Trigram_Text      = beam k=10 + trigram

Bu script tum anlamli paired McNemar karsilastirmalarini ureti:
  1. Greedy vs Greedy+Trigram   (LM katkisi)
  2. Beam vs Beam+Trigram       (Beam uzerinde LM katkisi)
  3. Greedy vs Beam             (decoder katkisi, LM yok)
  4. Greedy+Trigram vs Beam+Trigram  (decoder katkisi, LM var)
  5. Greedy vs Beam+Trigram     (en zayif vs en guclu)
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math

DEFAULT_CSV = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen\test_results_analysis.csv"
CSV = os.environ.get("CRNN_CSV", DEFAULT_CSV)

if not os.path.exists(CSV):
    print(f"[ERROR] CSV not found: {CSV}")
    sys.exit(1)

print(f"Using CSV: {CSV}\n")

rows = []
with open(CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

N = len(rows)
print(f"Total test samples: {N}\n")

# Helper functions
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def chi2_sf_1dof(x):
    return math.erfc(math.sqrt(x/2.0))

def binom_2sided_p(b, c):
    """Exact two-sided binomial p-value for McNemar."""
    n_disc = b + c
    k = min(b, c)
    if n_disc == 0:
        return 1.0
    s = 0.0
    for i in range(k+1):
        s += math.comb(n_disc, i) * (0.5 ** n_disc)
    return min(1.0, 2 * s)

def mcnemar(rows, col_before, col_after, label_before, label_after):
    """Run paired McNemar test between two predictor columns."""
    before_correct = []
    after_correct = []
    for r in rows:
        true = r["True_Text"]
        before_correct.append(r[col_before] == true)
        after_correct.append(r[col_after] == true)

    n = len(rows)
    n_b = sum(before_correct)
    n_a = sum(after_correct)
    acc_b = n_b / n
    acc_a = n_a / n
    delta_pp = (acc_a - acc_b) * 100

    # Contingency: b = before-correct & after-wrong, c = before-wrong & after-correct
    b = sum(1 for i in range(n) if before_correct[i] and not after_correct[i])
    c = sum(1 for i in range(n) if not before_correct[i] and after_correct[i])
    both = sum(1 for i in range(n) if before_correct[i] and after_correct[i])
    neither = sum(1 for i in range(n) if not before_correct[i] and not after_correct[i])

    # McNemar chi2 with continuity correction
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = 0.0
    p_chi = chi2_sf_1dof(chi2)
    p_exact = binom_2sided_p(b, c)

    ci_b = wilson_ci(n_b, n)
    ci_a = wilson_ci(n_a, n)

    print(f"\n{'='*72}")
    print(f"  {label_before}  vs  {label_after}")
    print(f"{'='*72}")
    print(f"  {label_before:35s}: {acc_b*100:6.2f}%   ({n_b:5d}/{n})  Wilson 95% CI [{ci_b[0]*100:.2f}, {ci_b[1]*100:.2f}]")
    print(f"  {label_after:35s}: {acc_a*100:6.2f}%   ({n_a:5d}/{n})  Wilson 95% CI [{ci_a[0]*100:.2f}, {ci_a[1]*100:.2f}]")
    print(f"  Delta:                                {delta_pp:+6.2f} pp")
    print(f"\n  Contingency table:")
    print(f"                              {label_after} correct  {label_after} wrong")
    print(f"  {label_before} correct       {both:5d}                {b:5d}")
    print(f"  {label_before} wrong         {c:5d}                {neither:5d}")
    print(f"\n  Discordant pairs: b+c = {b+c}")
    print(f"  McNemar chi-squared (continuity corrected): {chi2:.3f}")
    print(f"  Two-sided asymptotic p-value: {p_chi:.3e}")
    print(f"  Two-sided exact binomial p-value: {p_exact:.3e}")
    # Compact format for paper
    print(f"\n  PAPER FORMAT: WA={acc_b*100:.2f}% vs {acc_a*100:.2f}% "
          f"({delta_pp:+.2f}pp), chi2={chi2:.1f}, p_exact={p_exact:.2e}, "
          f"b={b}, c={c}")

# Check which columns exist
cols = list(rows[0].keys())
print("CSV columns:", cols)

has_beam = "Beam_Predicted_Text" in cols and "Beam_Trigram_Text" in cols

# ============================================================
# Comparison 1: Greedy vs Greedy + V2 Smart Trigram (ANA ABLATION)
# ============================================================
if "SmartTrigram_Text" in cols:
    mcnemar(rows, "Raw_Predicted_Text", "SmartTrigram_Text",
            "Greedy CTC (no LM)", "Greedy + V2 Smart Trigram")
else:
    mcnemar(rows, "Raw_Predicted_Text", "Predicted_Text",
            "Greedy CTC (no LM)", "Greedy + Trigram LM (v1)")

# ============================================================
# Comparison 1b: V1 (old loose) vs V2 (smart) Trigram - shows that the strategy matters
# ============================================================
if "SmartTrigram_Text" in cols:
    mcnemar(rows, "Predicted_Text", "SmartTrigram_Text",
            "V1 Loose Trigram (old)", "V2 Smart Trigram (new)")

if has_beam:
    # ========================================================
    # Comparison 2: Beam vs Beam + Smart Trigram
    # ========================================================
    beam_smart_col = "SmartTrigram_Beam_Text" if "SmartTrigram_Beam_Text" in cols else "Beam_Trigram_Text"
    mcnemar(rows, "Beam_Predicted_Text", beam_smart_col,
            "Beam Search k=10 (no LM)", "Beam k=10 + V2 Smart Trigram")

    # ========================================================
    # Comparison 3: Greedy vs Beam (decoding contribution, no LM)
    # ========================================================
    mcnemar(rows, "Raw_Predicted_Text", "Beam_Predicted_Text",
            "Greedy CTC (no LM)", "Beam Search k=10 (no LM)")

    # ========================================================
    # Comparison 4: Smart Greedy+Trigram vs Smart Beam+Trigram
    # ========================================================
    smart_g = "SmartTrigram_Text" if "SmartTrigram_Text" in cols else "Predicted_Text"
    mcnemar(rows, smart_g, beam_smart_col,
            "Greedy + Smart Trigram", "Beam + Smart Trigram")

    # ========================================================
    # Comparison 5: Raw Greedy (weakest) vs Beam+Smart Trigram (strongest)
    # ========================================================
    mcnemar(rows, "Raw_Predicted_Text", beam_smart_col,
            "Greedy CTC (weakest)", "Beam k=10 + Smart Trigram (strongest)")

# ============================================================
# V3 EXTENDED TRIGRAM (IAM + NLTK ~238K vocab) - HEADLINE ABLATION
# ============================================================
if "ExtendedTrigram_Text" in cols:
    print(f"\n\n{'#'*72}\n# V3 EXTENDED TRIGRAM (IAM + NLTK English wordlist, ~238K vocab)\n{'#'*72}")
    mcnemar(rows, "Raw_Predicted_Text", "ExtendedTrigram_Text",
            "Greedy CTC (no LM)", "Greedy + V3 Extended Trigram")

    if "SmartTrigram_Text" in cols:
        mcnemar(rows, "SmartTrigram_Text", "ExtendedTrigram_Text",
                "V2 Smart Trigram (IAM-only)", "V3 Extended Trigram (IAM+NLTK)")

    if "ExtendedTrigram_Beam_Text" in cols:
        mcnemar(rows, "Raw_Predicted_Text", "ExtendedTrigram_Beam_Text",
                "Greedy CTC (no LM, weakest)", "Beam + V3 Extended Trigram (strongest)")

# ============================================================
# Error analysis (full pipeline)
# ============================================================
print(f"\n{'='*72}")
print("  ERROR TYPE BREAKDOWN (Greedy + Trigram = full pipeline)")
print(f"{'='*72}")
from collections import Counter
errs = Counter(r["Error_Type"] for r in rows if r["Is_Correct"] == "False")
total_err = sum(errs.values())
for typ, n_err in errs.most_common():
    if typ == "None":
        continue
    print(f"  {typ:25s}: {n_err:5d} ({n_err/total_err*100:.1f}%)")
print(f"  TOTAL ERRORS              : {total_err}")

# Mean character accuracy
char_accs = [float(r["Character_Accuracy"]) for r in rows]
mean_ca = sum(char_accs) / len(char_accs)
print(f"\n  Mean Character Accuracy   : {mean_ca*100:.4f}%  (CER = {(1-mean_ca)*100:.4f}%)")
