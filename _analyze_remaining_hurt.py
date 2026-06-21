"""
V3 Extended Trigram'dan SONRA kalan hurt case'leri analiz eder.
Hangi tip kelimeler kaybedildi? Daha fazla vocab ekleme ne kadar yardim eder?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
from collections import Counter

CSV_PATH = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v2\test_results_analysis.csv"

# Load NLTK
import nltk
try:
    from nltk.corpus import words as nltk_words
    nltk_set = set(nltk_words.words())
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words
    nltk_set = set(nltk_words.words())

nltk_set_lower = {w.lower() for w in nltk_set}

# Load test CSV
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Find hurt cases AFTER V3 Extended Trigram
hurt_v3 = []
for r in rows:
    raw = r["Raw_Predicted_Text"]
    ext = r["ExtendedTrigram_Text"]
    true = r["True_Text"]
    if raw == true and ext != true:
        hurt_v3.append({
            "true": true,
            "raw_correct": raw,
            "ext_wrong": ext,
        })

print(f"Total hurt cases after V3 Extended: {len(hurt_v3)}")
print(f"(Down from 272 with V2 IAM-only trigram)")

# Categorize each hurt case
categories = {
    "Proper noun (CAP+...)": [],
    "Inflection (s/ed/ing)": [],
    "Compound or rare": [],
    "OCR artifact (1ye, b., etc.)": [],
    "Other valid English (missing from NLTK)": [],
}

for c in hurt_v3:
    true = c["true"]
    true_lower = true.lower()
    is_in_nltk = true_lower in nltk_set_lower

    # Categorize
    if not true[0].isalpha() or any(ch.isdigit() for ch in true) or true in ('.','-',','):
        categories["OCR artifact (1ye, b., etc.)"].append(c)
    elif true[0].isupper() and not is_in_nltk:
        categories["Proper noun (CAP+...)"].append(c)
    elif true_lower.endswith(('s','ed','ing','er')) and not is_in_nltk:
        # Check if base form is in NLTK
        base = true_lower
        for suff in ['s','ed','ing','er']:
            if base.endswith(suff):
                base_form = base[:-len(suff)]
                if base_form in nltk_set_lower or (base_form + 'e') in nltk_set_lower:
                    categories["Inflection (s/ed/ing)"].append(c)
                    break
        else:
            categories["Other valid English (missing from NLTK)"].append(c)
    elif is_in_nltk:
        # Strange - true IS in NLTK but still hurt? Probably edit distance issue
        categories["Other valid English (missing from NLTK)"].append(c)
    else:
        categories["Compound or rare"].append(c)

print("\n=== HURT CASE CATEGORIES ===")
for cat, items in categories.items():
    print(f"\n[{cat}]: {len(items)} cases")
    # Show first 5 examples
    for c in items[:5]:
        print(f"  TRUE='{c['true']:<20s}' RAW=correctly predicted, V3_corrected_to='{c['ext_wrong']}'")

# Stats
print("\n\n=== POTENTIAL IMPROVEMENT IF WE ADD MORE VOCAB ===")
print(f"Proper nouns       (need name database) : {len(categories['Proper noun (CAP+...)'])} cases ~ {len(categories['Proper noun (CAP+...)'])/5338*100:.2f}pp")
print(f"Inflections        (need morph forms)   : {len(categories['Inflection (s/ed/ing)'])} cases ~ {len(categories['Inflection (s/ed/ing)'])/5338*100:.2f}pp")
print(f"Other English      (more comprehensive) : {len(categories['Other valid English (missing from NLTK)'])} cases ~ {len(categories['Other valid English (missing from NLTK)'])/5338*100:.2f}pp")
print(f"Compound/rare      (specialized texts)  : {len(categories['Compound or rare'])} cases ~ {len(categories['Compound or rare'])/5338*100:.2f}pp")
print(f"OCR artifacts      (cannot fix vocab)   : {len(categories['OCR artifact (1ye, b., etc.)'])} cases")
print(f"\nTOTAL hurt: {len(hurt_v3)} cases = ~{len(hurt_v3)/5338*100:.2f}pp ceiling")

# Now also: how many OOV remain in test set that NLTK didn't catch?
test_words = set(r["True_Text"] for r in rows)
nltk_unknown_test = [w for w in test_words if w.lower() not in nltk_set_lower]
print(f"\n=== TEST SET COVERAGE BY NLTK ===")
print(f"Unique words in test  : {len(test_words)}")
print(f"In NLTK (lowercase)   : {len(test_words) - len(nltk_unknown_test)} ({(1-len(nltk_unknown_test)/len(test_words))*100:.1f}%)")
print(f"NOT in NLTK           : {len(nltk_unknown_test)} ({len(nltk_unknown_test)/len(test_words)*100:.1f}%)")

print(f"\nSample words NOT in NLTK (first 30):")
for w in sorted(nltk_unknown_test)[:30]:
    print(f"  {w}")
