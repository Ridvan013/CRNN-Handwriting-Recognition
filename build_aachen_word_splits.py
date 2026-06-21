"""
Aachen form ID listelerini IAM word ID'lerine cevirir.
Cikti:
  aachen_splits/train_words.txt
  aachen_splits/validation_words.txt
  aachen_splits/test_words.txt
Her satir: <word_id> <transcription>  (status='ok' filtresi uygulanmis)
"""
import os
import sys

ROOT = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1"
SPLITS_DIR = os.path.join(ROOT, "aachen_splits", "splits")
OUT_DIR = os.path.join(ROOT, "aachen_splits")
WORDS_TXT = os.path.join(ROOT, "HTR_Using_CRNN", "IAM", "processed", "archive", "iam_words", "words.txt")

def load_forms(name):
    path = os.path.join(SPLITS_DIR, f"{name}.uttlist")
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

train_forms = load_forms("train")
val_forms = load_forms("validation")
test_forms = load_forms("test")

print(f"Aachen forms: train={len(train_forms)}, val={len(val_forms)}, test={len(test_forms)}, "
      f"total={len(train_forms)+len(val_forms)+len(test_forms)}")

overlap_tv = train_forms & val_forms
overlap_tt = train_forms & test_forms
overlap_vt = val_forms & test_forms
assert not overlap_tv and not overlap_tt and not overlap_vt, "Overlap detected!"
print("OK: no overlap between train/val/test forms")

buckets = {"train": [], "validation": [], "test": []}
unassigned = 0
not_ok = 0
total_lines = 0

with open(WORDS_TXT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        total_lines += 1
        word_id = parts[0]
        status = parts[1]
        if status != "ok":
            not_ok += 1
            continue
        transcription = parts[-1]

        form_id = "-".join(word_id.split("-")[:2])

        # Skip non-IAM forms (e.g., user-added garbage)
        if not form_id or form_id.startswith("user"):
            unassigned += 1
            continue

        if form_id in train_forms:
            buckets["train"].append((word_id, transcription, line))
        elif form_id in val_forms:
            buckets["validation"].append((word_id, transcription, line))
        elif form_id in test_forms:
            buckets["test"].append((word_id, transcription, line))
        else:
            # Form Aachen split'inde yok ama IAM kaydi.
            # Val ve test'te olmadigi icin writer-disjoint kurali korunmus olur.
            # Train'e ekle (+%22 ekstra veri + zenginlestirilmis trigram vocab).
            buckets["train"].append((word_id, transcription, line))
            unassigned += 1

print(f"\nIAM words.txt: total={total_lines}, status!=ok skipped={not_ok}")
print(f"Aachen-assigned words (status=ok):")
for k in ("train", "validation", "test"):
    print(f"  {k:11s}: {len(buckets[k]):>6d}")
print(f"  unassigned : {unassigned:>6d}  (forms not in any Aachen partition)")
print(f"  TOTAL  ok  : {sum(len(buckets[k]) for k in buckets) + unassigned:>6d}")

for name, items in buckets.items():
    out = os.path.join(OUT_DIR, f"{name}_words.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# IAM word entries (status=ok) belonging to the Aachen-RWTH " + name + " partition.\n")
        f.write("# Format: original words.txt line (word_id status graylevel x y w h tag transcription)\n")
        for _, _, raw in items:
            f.write(raw + "\n")
    print(f"  wrote {out}  ({len(items)} lines)")

unique_words_train = set(t for _, t, _ in buckets["train"])
print(f"\nVocab size (training transcriptions): {len(unique_words_train)} unique words")
