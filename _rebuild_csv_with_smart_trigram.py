"""
Mevcut Model_aachen/test_results_analysis.csv'yi yenilenmiş V2 trigram ile
yeniden işler. Eski sütunları korur, "SmartTrigram_Text" sütunu ekler.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import os

sys.path.insert(0, r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1")
from trigram_lm import TrigramLanguageModel

CSV_IN = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen\test_results_analysis.csv"
CSV_OUT = CSV_IN  # in-place update
WORDS_FILE = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt"

print("Loading V2 trigram LM...")
lm = TrigramLanguageModel(WORDS_FILE)

print("Reading existing CSV...")
with open(CSV_IN, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = reader.fieldnames

print(f"Total samples: {len(rows)}")
print("Applying V2 (smart) trigram correction to Raw_Predicted_Text and Beam_Predicted_Text...")

helped_g = 0; hurt_g = 0; touched_g = 0
helped_b = 0; hurt_b = 0; touched_b = 0
for r in rows:
    raw = r["Raw_Predicted_Text"]
    true = r["True_Text"]
    new_text = lm.correct_word(raw)
    r["SmartTrigram_Text"] = new_text
    if new_text != raw:
        touched_g += 1
        if raw == true and new_text != true:
            hurt_g += 1
        elif raw != true and new_text == true:
            helped_g += 1

    if "Beam_Predicted_Text" in r:
        beam = r["Beam_Predicted_Text"]
        new_beam = lm.correct_word(beam)
        r["SmartTrigram_Beam_Text"] = new_beam
        if new_beam != beam:
            touched_b += 1
            if beam == true and new_beam != true:
                hurt_b += 1
            elif beam != true and new_beam == true:
                helped_b += 1

# Update fields list
new_fields = list(fields)
for nf in ["SmartTrigram_Text", "SmartTrigram_Beam_Text"]:
    if nf not in new_fields:
        new_fields.append(nf)

print(f"\nSmart Trigram on Greedy:")
print(f"  Touched: {touched_g}, Helped: {helped_g}, Hurt: {hurt_g}, Net: {helped_g - hurt_g:+d}")
print(f"\nSmart Trigram on Beam:")
print(f"  Touched: {touched_b}, Helped: {helped_b}, Hurt: {hurt_b}, Net: {helped_b - hurt_b:+d}")

# Compute WAs
raw_correct = sum(1 for r in rows if r["Raw_Predicted_Text"] == r["True_Text"])
smart_correct = sum(1 for r in rows if r["SmartTrigram_Text"] == r["True_Text"])
old_correct = sum(1 for r in rows if r["Predicted_Text"] == r["True_Text"])

print(f"\nWord Accuracy Summary (5,338 Aachen test samples):")
print(f"  Raw Greedy             : {raw_correct/len(rows)*100:6.2f}%  ({raw_correct}/{len(rows)})")
print(f"  OLD Trigram (v1, loose): {old_correct/len(rows)*100:6.2f}%  ({old_correct}/{len(rows)})  Delta vs raw: {(old_correct-raw_correct)/len(rows)*100:+.2f}pp")
print(f"  V2 Smart Trigram       : {smart_correct/len(rows)*100:6.2f}%  ({smart_correct}/{len(rows)})  Delta vs raw: {(smart_correct-raw_correct)/len(rows)*100:+.2f}pp")

# Write back
print(f"\nWriting updated CSV to: {CSV_OUT}")
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    for r in rows:
        # Ensure all fields present
        for nf in new_fields:
            r.setdefault(nf, "")
        writer.writerow(r)

print("Done.")
