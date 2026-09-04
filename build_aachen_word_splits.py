"""
Aachen form ID listelerini IAM word ID'lerine cevirir.
Cikti:
  aachen_splits/train_words.txt
  aachen_splits/validation_words.txt
  aachen_splits/test_words.txt
Her satir: <word_id> <transcription>  (status='ok' filtresi uygulanmis)
"""
import os
import re
import sys
import argparse

_ap = argparse.ArgumentParser(description="IAM words.txt -> Aachen word-level split dosyalari")
_ap.add_argument("--words-txt", default=None, help="IAM words.txt yolu (varsayilan: repo icindeki)")
_ap.add_argument("--keep-val-test-text-overlap", action="store_true",
                 help="Metni test kumesinde de gecen dogrulama formlarini TUT. "
                      "VARSAYILAN KAPALI: bu formlar dogrulamadan cikarilir ki en "
                      "iyi epoch secimi test kumesinden bagimsiz olsun. Test kumesine DOKUNULMAZ.")
_ap.add_argument("--include-unassigned", action="store_true",
                 help="Aachen'in hicbir bolumunde olmayan 340 formu egitime EKLE. "
                      "VARSAYILAN KAPALI: yazar-ayriklik garantisi ve literaturle "
                      "karsilastirilabilirlik icin yalniz resmi 747/116/336 form kullanilir.")
_args = _ap.parse_args()

ROOT = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1"
SPLITS_DIR = os.path.join(ROOT, "aachen_splits", "splits")
OUT_DIR = os.path.join(ROOT, "aachen_splits")
WORDS_TXT = _args.words_txt or os.path.join(ROOT, "HTR_Using_CRNN", "IAM", "processed", "archive", "iam_words", "words.txt")
print(f"words.txt: {WORDS_TXT}")
print(f"mode     : {'include-unassigned (train += 340 extra forms)' if _args.include_unassigned else 'STRICT (official Aachen forms only)'}")

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
            # Form, Aachen'in hicbir bolumunde yok (IAM'deki 1539 formun 340'i).
            # Bu formlarin yazarlari test/val yazarlariyla cakisabilir; forms.txt
            # olmadan kontrol edilemez. Bu yuzden VARSAYILAN olarak atlanir.
            unassigned += 1
            if _args.include_unassigned:
                buckets["train"].append((word_id, transcription, line))

# ---------------------------------------------------------------------------
# Dogrulama <-> test METIN ortusmesinin temizlenmesi
# IAM form id'si <korpus><no>-<metin><varyant> seklinde (ornek f07-028a). Ayni
# metnin farkli varyantlari resmi bolmede val ve test'e dagilmis olabilir
# (f07-028b val'de, f07-028a test'te). En iyi epoch'u val'de sectigimiz icin bu
# durum secimi test'e karsi hafif yanli hale getirebilir. Bu formlari
# DOGRULAMADAN cikariyoruz. Test kumesine dokunulmaz; egitime de tasinmaz
# (tasinsa bu kez train<->test metin ortusmesi dogar).
# ---------------------------------------------------------------------------
def _text_base(form_id):
    m = re.match(r"^([a-z]\d+-\d+)", form_id)
    return m.group(1) if m else form_id

_form_of = lambda wid: "-".join(wid.split("-")[:2])
_test_bases = {_text_base(_form_of(w)) for w, _, _ in buckets["test"]}
_before_n = len(buckets["validation"])
_dropped = sorted({_form_of(w) for w, _, _ in buckets["validation"]
                   if _text_base(_form_of(w)) in _test_bases})
if _dropped and not _args.keep_val_test_text_overlap:
    buckets["validation"] = [r for r in buckets["validation"]
                             if _form_of(r[0]) not in _dropped]
    _n = _before_n - len(buckets["validation"])
    print("")
    print(f"val<->test metin ortusmesi: {len(_dropped)} form dogrulamadan cikarildi ({_n} kelime)")
    print(f"  {chr(39).join([])}{', '.join(_dropped)}")
    print("  test kumesi DEGISTIRILMEDI")
elif _dropped:
    print(f"val<->test metin ortusmesi TUTULDU: {len(_dropped)} form")
else:
    print("val<->test metin ortusmesi: yok")

print(f"\nIAM words.txt: total={total_lines}, status!=ok skipped={not_ok}")
print(f"Aachen-assigned words (status=ok):")
for k in ("train", "validation", "test"):
    print(f"  {k:11s}: {len(buckets[k]):>6d}")
print(f"  unassigned : {unassigned:>6d}  (forms not in any Aachen partition)")
print(f"  TOTAL used : {sum(len(buckets[k]) for k in buckets):>6d}")
# form sayilari (gercekten etiketi olan)
for k in ("train", "validation", "test"):
    forms_seen = {"-".join(w.split("-")[:2]) for w, _, _ in buckets[k]}
    print(f"  {k:11s} forms with labels: {len(forms_seen)}")

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
