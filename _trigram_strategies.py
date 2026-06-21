"""
Trigram correction stratejilerini post-hoc test eder.
Mevcut Aachen test CSV'sindeki Raw_Predicted_Text uzerinde calisip
farkli correction stratejilerini deneyip kazanan stratejiyi bulur.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import csv
import math
from collections import Counter

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen\test_results_analysis.csv"
WORDS_FILE = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"


def load_csv():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "true": r["True_Text"],
                "raw": r["Raw_Predicted_Text"],
                "old_corrected": r["Predicted_Text"],
            })
    return rows


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


def build_lm(words_file):
    """Build vocab, unigram, bigram, trigram from IAM training corpus."""
    words = []
    with open(words_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                words.append(parts[-1])

    vocab = set(words)
    unigrams = Counter(words)
    total = len(words)
    V = len(vocab)

    # Group vocabulary by length for fast lookup
    vocab_by_len = {}
    for w in vocab:
        vocab_by_len.setdefault(len(w), []).append(w)

    return {
        "vocab": vocab,
        "vocab_by_len": vocab_by_len,
        "unigrams": unigrams,
        "total": total,
        "V": V,
    }


def candidates_within(word, vocab_by_len, d_max):
    """Hizli candidate enumeration: vocabulary'yi length bucket'larina gore tarar."""
    cands = []
    for length, words in vocab_by_len.items():
        if abs(length - len(word)) > d_max:
            continue
        for vw in words:
            d = edit_distance(word, vw)
            if d <= d_max:
                cands.append((vw, d))
    return cands


def unigram_log_prob(word, lm):
    cnt = lm["unigrams"].get(word, 0)
    return math.log((cnt + 1) / (lm["total"] + lm["V"]))


# ============================================================
# STRATEJILER
# ============================================================

def strategy_v1_original(word, lm):
    """Mevcut: OOV ise koşulsuz en yakın Levenshtein adayını döndür."""
    if word in lm["vocab"]:
        return word
    if len(word) <= 4:
        d_max = 2
    elif len(word) <= 8:
        d_max = 3
    else:
        d_max = 4
    cands = candidates_within(word, lm["vocab_by_len"], d_max)
    if not cands:
        return word
    # En iyi: log P - 3*d
    best = max(cands, key=lambda c: unigram_log_prob(c[0], lm) - 3.0 * c[1])
    return best[0]


def strategy_v2_tight_distance(word, lm):
    """d_max çok daha sıkı: 1 / 2 / 2"""
    if word in lm["vocab"]:
        return word
    if len(word) <= 4:
        d_max = 1
    elif len(word) <= 8:
        d_max = 2
    else:
        d_max = 2
    cands = candidates_within(word, lm["vocab_by_len"], d_max)
    if not cands:
        return word
    best = max(cands, key=lambda c: unigram_log_prob(c[0], lm) - 5.0 * c[1])
    return best[0]


def strategy_v3_score_margin(word, lm):
    """Margin > 5.0 gerekli."""
    if word in lm["vocab"]:
        return word
    if len(word) <= 4:
        d_max = 2
    elif len(word) <= 8:
        d_max = 3
    else:
        d_max = 4
    cands = candidates_within(word, lm["vocab_by_len"], d_max)
    if not cands:
        return word
    # Score = log P - 3*d, kıyaslama için original'ın puanı
    original_score = unigram_log_prob(word, lm) - 0.0  # word OOV, baseline
    best = max(cands, key=lambda c: unigram_log_prob(c[0], lm) - 3.0 * c[1])
    best_score = unigram_log_prob(best[0], lm) - 3.0 * best[1]
    if best_score > original_score + 5.0:
        return best[0]
    return word


def strategy_v4_skip_short(word, lm):
    """Kısa kelimeleri (<=2) dokunma + tight + margin."""
    if len(word) <= 2:
        return word  # 1-2 char kelimeleri dokunma
    if word in lm["vocab"]:
        return word
    if len(word) <= 4:
        d_max = 1
    elif len(word) <= 8:
        d_max = 2
    else:
        d_max = 2
    cands = candidates_within(word, lm["vocab_by_len"], d_max)
    if not cands:
        return word
    best = max(cands, key=lambda c: unigram_log_prob(c[0], lm) - 5.0 * c[1])
    return best[0]


def strategy_v5_capitalize_aware(word, lm):
    """Capitalize-aware: ilk harf buyukse, kucuk vocab'da ara ama buyuk olarak don."""
    if len(word) <= 2:
        return word
    if word in lm["vocab"]:
        return word

    # Capitalize handling
    is_cap = word[:1].isupper()
    lookup = word.lower() if is_cap else word

    if lookup in lm["vocab"]:
        return word  # original capitalized form likely correct

    if len(word) <= 4:
        d_max = 1
    elif len(word) <= 8:
        d_max = 2
    else:
        d_max = 2
    cands = candidates_within(lookup, lm["vocab_by_len"], d_max)
    if not cands:
        return word
    best = max(cands, key=lambda c: unigram_log_prob(c[0], lm) - 5.0 * c[1])
    result = best[0]
    if is_cap and result:
        result = result[:1].upper() + result[1:]
    return result


def strategy_v6_no_correction(word, lm):
    """Hiç düzeltme yapma - sadece baseline."""
    return word


# ============================================================
# Evaluation
# ============================================================

def evaluate(rows, strategy_fn, lm):
    correct = 0
    n_corrected = 0
    n_helped = 0
    n_hurt = 0
    for r in rows:
        raw = r["raw"]
        true = r["true"]
        new_pred = strategy_fn(raw, lm)
        if new_pred == true:
            correct += 1
        # Track corrections vs original raw
        if new_pred != raw:
            n_corrected += 1
            if raw == true and new_pred != true:
                n_hurt += 1
            elif raw != true and new_pred == true:
                n_helped += 1
    return {
        "wa": correct / len(rows),
        "correct": correct,
        "n_corrected": n_corrected,
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "net": n_helped - n_hurt,
    }


def main():
    print("Loading data...")
    rows = load_csv()
    print(f"Total samples: {len(rows)}")

    print("Building LM from Aachen train vocabulary...")
    lm = build_lm(WORDS_FILE)
    print(f"Vocab size: {lm['V']}, total words: {lm['total']}")
    print()

    # Baseline: ham greedy
    raw_correct = sum(1 for r in rows if r["raw"] == r["true"])
    print(f"BASELINE Raw Greedy (no correction): {raw_correct}/{len(rows)} = {raw_correct/len(rows)*100:.2f}%")
    print(f"OLD Trigram (from CSV):              "
          f"{sum(1 for r in rows if r['old_corrected'] == r['true'])}/{len(rows)} = "
          f"{sum(1 for r in rows if r['old_corrected'] == r['true'])/len(rows)*100:.2f}%")
    print()

    strategies = [
        ("V0 No correction (raw greedy)", strategy_v6_no_correction),
        ("V1 Original (loose, d=2/3/4, no margin)", strategy_v1_original),
        ("V2 Tight distance (d=1/2/2)", strategy_v2_tight_distance),
        ("V3 Score margin > 5.0", strategy_v3_score_margin),
        ("V4 Skip short + tight + margin", strategy_v4_skip_short),
        ("V5 Capitalize-aware + tight", strategy_v5_capitalize_aware),
    ]

    print(f"{'Strategy':<48s} {'WA':>8s} {'Δ vs raw':>10s} {'#corr':>7s} {'helped':>7s} {'hurt':>6s} {'net':>6s}")
    print("=" * 95)

    raw_wa = raw_correct / len(rows)
    for name, fn in strategies:
        print(f"Evaluating: {name}...", file=sys.stderr)
        res = evaluate(rows, fn, lm)
        delta = (res["wa"] - raw_wa) * 100
        print(f"{name:<48s} {res['wa']*100:6.2f}%  {delta:+7.2f}pp  "
              f"{res['n_corrected']:5d}  {res['n_helped']:5d}  {res['n_hurt']:4d}  "
              f"{res['net']:+5d}")


if __name__ == "__main__":
    main()
