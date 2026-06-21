"""
Apply Extended Trigram (IAM+NLTK) to Beam column too,
giving us all 5 configurations for the article.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
from collections import Counter

import nltk
try:
    from nltk.corpus import words as nltk_words_corpus
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words_corpus

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v2\test_results_analysis.csv"
TRAIN_WORDS_FILE = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"

iam_vocab = set()
unigram = Counter()
with open(TRAIN_WORDS_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 9:
            w = parts[-1]
            iam_vocab.add(w)
            unigram[w] += 1

nltk_vocab = set(nltk_words_corpus.words())
extended_vocab = iam_vocab | nltk_vocab
extended_vocab_lower = {w.lower() for w in extended_vocab}

vocab_by_len = {}
for w in extended_vocab:
    vocab_by_len.setdefault(len(w), []).append(w)

total_iam = sum(unigram.values())
V = len(iam_vocab)


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def correct_extended(word):
    if word in extended_vocab or word.lower() in extended_vocab_lower:
        return word

    if len(word) <= 4:
        d_max = 1
    elif len(word) <= 8:
        d_max = 2
    else:
        d_max = 2

    cands = []
    for length, ws in vocab_by_len.items():
        if abs(length - len(word)) > d_max:
            continue
        for vw in ws:
            d = edit_distance(word, vw)
            if d <= d_max:
                if vw in unigram:
                    score = math.log((unigram[vw] + 1) / (total_iam + V)) - d * 5.0
                else:
                    score = math.log(0.5 / (total_iam + V)) - d * 5.0
                cands.append((vw, score))
    if cands:
        cands.sort(key=lambda c: c[1], reverse=True)
        return cands[0][0]
    return word


# Load CSV
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
fields = list(rows[0].keys())

print(f"Applying Extended Trigram to Beam column ({len(rows)} samples)...")
for r in rows:
    beam = r["Beam_Predicted_Text"]
    r["ExtendedTrigram_Beam_Text"] = correct_extended(beam)

if "ExtendedTrigram_Beam_Text" not in fields:
    fields.append("ExtendedTrigram_Beam_Text")

# Summary
N = len(rows)
raw_g = sum(1 for r in rows if r["Raw_Predicted_Text"] == r["True_Text"])
raw_b = sum(1 for r in rows if r["Beam_Predicted_Text"] == r["True_Text"])
ext_g = sum(1 for r in rows if r["ExtendedTrigram_Text"] == r["True_Text"])
ext_b = sum(1 for r in rows if r["ExtendedTrigram_Beam_Text"] == r["True_Text"])

print(f"\n{'='*72}")
print(f"  V2 MODEL ON AACHEN TEST SET ({N} samples) - ALL CONFIGURATIONS")
print(f"{'='*72}")
print(f"  Raw Greedy                     : {raw_g:5d}/{N}  =  {raw_g/N*100:6.2f}%")
print(f"  Raw Beam k=10                  : {raw_b:5d}/{N}  =  {raw_b/N*100:6.2f}%")
print(f"  Greedy + Extended Trigram      : {ext_g:5d}/{N}  =  {ext_g/N*100:6.2f}%  ({(ext_g-raw_g)/N*100:+.2f}pp vs greedy)")
print(f"  Beam + Extended Trigram        : {ext_b:5d}/{N}  =  {ext_b/N*100:6.2f}%  ({(ext_b-raw_g)/N*100:+.2f}pp vs greedy)")

# McNemar Raw vs Beam+Extended
b_corr = [r["Raw_Predicted_Text"] == r["True_Text"] for r in rows]
e_corr = [r["ExtendedTrigram_Beam_Text"] == r["True_Text"] for r in rows]
b = sum(1 for i in range(N) if b_corr[i] and not e_corr[i])
c = sum(1 for i in range(N) if not b_corr[i] and e_corr[i])
chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
p_chi = math.erfc(math.sqrt(chi2 / 2.0))
n_disc = b + c
k = min(b, c)
p_exact = 2 * sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
p_exact = min(p_exact, 1.0)
print(f"\n  McNemar (Raw Greedy vs Beam + Extended Trigram):")
print(f"    chi2={chi2:.2f}, p_asymp={p_chi:.3e}, p_exact={p_exact:.3e}, b={b}, c={c}")

# Wilson CI for best
def wilson(k, n, z=1.96):
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - margin, center + margin)

ci_best = wilson(ext_b, N)
print(f"\n  Best config (Beam + Extended): Wilson 95% CI [{ci_best[0]*100:.2f}, {ci_best[1]*100:.2f}]")

# Write back
with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print(f"\nCSV updated: {CSV_PATH}")
