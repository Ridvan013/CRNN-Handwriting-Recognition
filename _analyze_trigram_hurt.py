"""
Trigram'in yanlislikla bozdugu kelimeleri analiz eder.
V2 sonucundaki 272 hurt case'ine bakar.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
from collections import Counter

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v2\test_results_analysis.csv"
TRAIN_WORDS_FILE = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"

# Load training vocab
train_vocab = set()
with open(TRAIN_WORDS_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 9:
            train_vocab.add(parts[-1])
print(f"Training vocabulary size: {len(train_vocab)}")

# Read CSV
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"Total test samples: {len(rows)}")

# Find hurt cases: Raw_Predicted correct but Predicted_Text (smart trigram) wrong
hurt_cases = []
helped_cases = []
for r in rows:
    raw = r["Raw_Predicted_Text"]
    smart = r["Predicted_Text"]
    true = r["True_Text"]
    if raw == true and smart != true:
        hurt_cases.append({
            "true": true,
            "raw_correct": raw,
            "smart_wrong": smart,
            "true_in_train": true in train_vocab,
            "smart_in_train": smart in train_vocab,
        })
    elif raw != true and smart == true:
        helped_cases.append({
            "true": true,
            "raw_wrong": raw,
            "smart_correct": smart,
        })

print(f"\nHurt cases (Raw correct -> Smart wrong): {len(hurt_cases)}")
print(f"Helped cases (Raw wrong -> Smart correct): {len(helped_cases)}")

# Of hurt cases, how many had the TRUE word NOT in training vocab?
hurt_oov = [c for c in hurt_cases if not c["true_in_train"]]
hurt_inv = [c for c in hurt_cases if c["true_in_train"]]
print(f"\nOf {len(hurt_cases)} hurt cases:")
print(f"  TRUE word IS in train vocab: {len(hurt_inv)} ({len(hurt_inv)/len(hurt_cases)*100:.1f}%)")
print(f"  TRUE word NOT in train vocab: {len(hurt_oov)} ({len(hurt_oov)/len(hurt_cases)*100:.1f}%)")
print(f"  -> {len(hurt_oov)} cases are 'we corrected a valid English word that we just didnt have in our dict'")

print(f"\nSample hurt cases (first 20 - true word NOT in train vocab):")
print(f"  {'TRUE':<20s} {'RAW (correct)':<20s} {'SMART (wrong)':<20s}")
for c in hurt_oov[:20]:
    print(f"  {c['true']:<20s} {c['raw_correct']:<20s} {c['smart_wrong']:<20s}")

print(f"\nSample hurt cases - TRUE word IS in train vocab (rare but interesting):")
print(f"  {'TRUE':<20s} {'RAW (correct)':<20s} {'SMART (wrong)':<20s}")
for c in hurt_inv[:10]:
    print(f"  {c['true']:<20s} {c['raw_correct']:<20s} {c['smart_wrong']:<20s}")

# Count words in test set NOT in training vocab
test_words = [r["True_Text"] for r in rows]
test_unique = set(test_words)
oov_in_test = test_unique - train_vocab
print(f"\n=== Aachen Test Vocabulary Analysis ===")
print(f"Unique words in test: {len(test_unique)}")
print(f"In training vocab: {len(test_unique & train_vocab)} ({len(test_unique & train_vocab)/len(test_unique)*100:.1f}%)")
print(f"NOT in training vocab: {len(oov_in_test)} ({len(oov_in_test)/len(test_unique)*100:.1f}%)")
print(f"\nTotal OOV occurrences in test set: {sum(1 for w in test_words if w not in train_vocab)}/{len(test_words)} ({sum(1 for w in test_words if w not in train_vocab)/len(test_words)*100:.1f}%)")

# Top OOV words by frequency
oov_freq = Counter(w for w in test_words if w not in train_vocab)
print(f"\nTop 30 most frequent OOV words in test set (model COULD predict these correctly,")
print(f"but trigram would 'correct' them since they're OOV):")
for w, c in oov_freq.most_common(30):
    print(f"  {w:<25s} appears {c}x")
