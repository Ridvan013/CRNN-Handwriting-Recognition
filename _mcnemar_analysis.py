"""
McNemar testi + Wilson 95% guven araliklari
test_results_analysis.csv uzerinden:
  - Raw_Predicted_Text  -> GREEDY decode (greedy.py:1559 greedy_decode)
  - Predicted_Text      -> GREEDY + Trigram LM (greedy.py:1586 correct_word)

CSV path env degiskeniyle ovveride edilebilir:
  $env:CRNN_CSV = "C:\\path\\to\\test_results_analysis.csv"; python _mcnemar_analysis.py

Varsayilan: Aachen sonuclari (Model_aachen). Eski custom split icin Model klasorune yonlendir.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
from collections import Counter

# Default: Aachen split (Model_aachen). CRNN_CSV env ile override edilebilir.
DEFAULT_CSV = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen\test_results_analysis.csv"
CSV = os.environ.get("CRNN_CSV", DEFAULT_CSV)

if not os.path.exists(CSV):
    # Fallback: try old custom-split CSV
    fallback = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model\test_results_analysis.csv"
    if os.path.exists(fallback):
        print(f"[Warning] Aachen CSV not found at {CSV}", file=sys.stderr)
        print(f"[Warning] Falling back to old custom-split CSV: {fallback}", file=sys.stderr)
        CSV = fallback
    else:
        print(f"[ERROR] CSV not found: {CSV}", file=sys.stderr)
        print("Train the Aachen model first (greedy_aachen.py), or set $env:CRNN_CSV", file=sys.stderr)
        sys.exit(1)

print(f"Using CSV: {CSV}")

rows = []
with open(CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

N = len(rows)
print(f"Toplam örnek: {N}")

# Per-sample doğruluk
beam_correct = [r["Raw_Predicted_Text"] == r["True_Text"] for r in rows]
full_correct = [r["Predicted_Text"] == r["True_Text"] for r in rows]

acc_beam = sum(beam_correct) / N
acc_full = sum(full_correct) / N
print(f"Beam Search (LM öncesi) WA = {acc_beam*100:.2f}%  ({sum(beam_correct)}/{N})")
print(f"Full pipeline (Beam+LM)  WA = {acc_full*100:.2f}%  ({sum(full_correct)}/{N})")
print(f"Δ WA = {(acc_full-acc_beam)*100:+.2f} pp")

# McNemar contingency table
b = sum(1 for i in range(N) if beam_correct[i] and not full_correct[i])  # beam ✓, full ✗
c = sum(1 for i in range(N) if not beam_correct[i] and full_correct[i])  # beam ✗, full ✓
both = sum(1 for i in range(N) if beam_correct[i] and full_correct[i])
neither = sum(1 for i in range(N) if not beam_correct[i] and not full_correct[i])

print("\n=== McNemar Kontenjans Tablosu (Beam vs Full) ===")
print(f"                     Full DOĞRU  Full YANLIŞ")
print(f"Beam DOĞRU            {both:5d}        {b:5d}")
print(f"Beam YANLIŞ           {c:5d}        {neither:5d}")
print(f"\nb (beam-only doğru) = {b}")
print(f"c (LM-sonrası kazanılan) = {c}")
print(f"Discordant pairs: b+c = {b+c}")

# Exact McNemar (binomial test) - tek yönlü ve iki yönlü
# Approx chi-square with continuity correction
if (b + c) > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
else:
    chi2 = 0.0

# p-value from chi-square (1 dof)
# Use survival function approximation
def chi2_sf_1dof(x):
    # erfc-based survival for chi2 with 1 dof: P(X>x) = erfc(sqrt(x/2))
    return math.erfc(math.sqrt(x/2.0))

p_chi = chi2_sf_1dof(chi2)
print(f"\nMcNemar χ² (with continuity correction) = {chi2:.3f}")
print(f"p-value (asymptotic, two-sided) = {p_chi:.3e}")

# Exact binomial two-sided p-value
def binom_cdf(k, n, p=0.5):
    """ P(X <= k) for Binomial(n, p) """
    s = 0.0
    for i in range(k+1):
        s += math.comb(n, i) * (p**i) * ((1-p)**(n-i))
    return s

n_disc = b + c
k = min(b, c)
if n_disc > 0:
    p_exact = 2 * binom_cdf(k, n_disc, 0.5)
    p_exact = min(p_exact, 1.0)
else:
    p_exact = 1.0
print(f"Exact McNemar (binomial) p-value (two-sided) = {p_exact:.3e}")

# Wilson 95% CI
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - margin, center + margin)

ci_beam = wilson_ci(sum(beam_correct), N)
ci_full = wilson_ci(sum(full_correct), N)
print(f"\n=== Wilson 95% Güven Aralıkları ===")
print(f"Beam:  {acc_beam*100:.2f}%  [{ci_beam[0]*100:.2f}%, {ci_beam[1]*100:.2f}%]")
print(f"Full:  {acc_full*100:.2f}%  [{ci_full[0]*100:.2f}%, {ci_full[1]*100:.2f}%]")

# Two-proportion z-test (independent assumption - daha gevşek; McNemar tercih)
p_pool = (sum(beam_correct) + sum(full_correct)) / (2*N)
se = math.sqrt(p_pool*(1-p_pool)*(2.0/N))
z = (acc_full - acc_beam) / se if se > 0 else 0.0
p_z = 2 * (1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
print(f"\nİki-orantı z-testi (independent): z={z:.3f}, p={p_z:.3e}")

# Error analysis - LM correction yardım etti vs zarar verdi
lm_helped = c  # beam yanlış, LM düzeltti
lm_hurt = b    # beam doğru, LM bozdu
print(f"\n=== Trigram LM Etki Analizi ===")
print(f"LM doğru düzeltti (gained): {lm_helped}")
print(f"LM yanlış düzeltti (hurt): {lm_hurt}")
print(f"Net kazanç: {lm_helped - lm_hurt}")
print(f"Düzeltme tetiklenen örnek: {sum(1 for r in rows if r['Raw_Predicted_Text'] != r['Predicted_Text'])}")

# Error type breakdown
err_types = Counter(r['Error_Type'] for r in rows if r['Is_Correct'] == 'False')
print(f"\n=== Hata tipi dağılımı ===")
for t, n in err_types.most_common():
    print(f"  {t}: {n} ({n/sum(err_types.values())*100:.1f}%)")
