#!/usr/bin/env python3
"""
Aachen word-level split dogrulama scripti.

Makalede iddia edilen her yapisal ozelligi bagimsiz olarak kontrol eder:
form ayrikligi, YAZAR ayrikligi, resmi uttlist ile birebir eslesme,
metin (prompt) ayrikligi, goruntu butunlugu.

Calistirma:
    python verify_aachen_splits.py
Cikis kodu 0 = hepsi gecti, 1 = en az bir kontrol basarisiz.
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(ROOT, "aachen_splits")
# Goruntu dizini: --img-root ile veya IAM_ROOT ortam degiskeniyle ezilebilir
# (Kaggle'da goruntuler /kaggle/input altinda durur).
IMG_ROOT = os.path.join(ROOT, "HTR_Using_CRNN", "IAM", "processed",
                        "archive", "iam_words", "words")
for _i, _a in enumerate(sys.argv):
    if _a == "--img-root" and _i + 1 < len(sys.argv):
        IMG_ROOT = sys.argv[_i + 1]
IMG_ROOT = os.environ.get("IAM_ROOT", IMG_ROOT)

FILES = {"train": "train_words.txt",
         "val": "validation_words.txt",
         "test": "test_words.txt"}
UTT = {"train": "train.uttlist",
       "val": "validation.uttlist",
       "test": "test.uttlist"}

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'GECTI' if ok else 'KALDI'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, detail=""):
    """Ortamda yapilamayan kontrol; basarisizlik sayilmaz."""
    print(f"  [ATLA ] {name}" + (f"  {detail}" if detail else ""))


def form_of(word_id):
    return "-".join(word_id.split("-")[:2])


def text_base(form_id):
    m = re.match(r"^([a-z]\d+-\d+)", form_id)
    return m.group(1) if m else form_id


def load_records(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split())
    return rows


def main():
    recs = {k: load_records(os.path.join(SPLIT_DIR, v)) for k, v in FILES.items()}
    forms = {k: {form_of(r[0]) for r in v} for k, v in recs.items()}
    utt = {}
    for k, v in UTT.items():
        with open(os.path.join(SPLIT_DIR, "splits", v)) as fh:
            utt[k] = {l.strip() for l in fh if l.strip()}

    print("\nBOLUM BUYUKLUKLERI")
    for k in ("train", "val", "test"):
        print(f"  {k:<6} {len(forms[k]):>4} form  {len(recs[k]):>7,} kelime")

    print("\n1. FORM AYRIKLIGI")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = forms[a] & forms[b]
        check(f"{a} vs {b}", not ov, f"ortak {len(ov)}")

    print("\n2. RESMI LISTEYLE ESLESME")
    check("test resmi listenin TAMAMI", forms["test"] == utt["test"],
          f"{len(forms['test'])}/{len(utt['test'])}")
    check("train resmi listenin TAMAMI", forms["train"] == utt["train"],
          f"{len(forms['train'])}/{len(utt['train'])}")
    check("val resmi listenin ALT KUMESI", forms["val"] <= utt["val"],
          f"{len(forms['val'])}/{len(utt['val'])} "
          f"({len(utt['val'] - forms['val'])} form metin ortusmesi nedeniyle cikarildi)")

    print("\n3. YAZAR AYRIKLIGI")
    fw_path = os.path.join(SPLIT_DIR, "form_writer.txt")
    if not os.path.exists(fw_path):
        check("form_writer.txt mevcut", False, "dosya yok, yazar kontrolu ATLANDI")
    else:
        f2w = {}
        with open(fw_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    a, b = line.split()
                    f2w[a] = b
        unknown = [f for v in forms.values() for f in v if f not in f2w]
        check("tum formlarin yazari biliniyor", not unknown, f"eksik {len(unknown)}")
        writers = {k: {f2w[f] for f in v if f in f2w} for k, v in forms.items()}
        for k in ("train", "val", "test"):
            print(f"         {k:<6} {len(writers[k]):>4} yazar")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = writers[a] & writers[b]
            check(f"{a} vs {b}", not ov,
                  f"ortak {len(ov)}" + (f" -> {sorted(ov)[:5]}" if ov else ""))

    print("\n4. METIN (PROMPT) AYRIKLIGI")
    bases = {k: {text_base(f) for f in v} for k, v in forms.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = bases[a] & bases[b]
        check(f"{a} vs {b}", not ov, f"ortak {len(ov)}")

    print("\n5. KAYIT BUTUNLUGU")
    bad_status = sum(1 for v in recs.values() for r in v if r[1] != "ok")
    check("hepsi status=ok", bad_status == 0, f"ihlal {bad_status}")
    ids = [r[0] for v in recs.values() for r in v]
    check("tekrar eden word_id yok", len(ids) == len(set(ids)),
          f"tekrar {len(ids) - len(set(ids))}")
    junk = sum(1 for v in recs.values() for r in v if not re.match(r"^[a-z]\d+-", r[0]))
    check("IAM disi kayit yok", junk == 0, f"bulunan {junk}")
    wrong = sum(1 for k, v in recs.items() for r in v if form_of(r[0]) not in utt[k])
    check("her kayit dogru bolumde", wrong == 0, f"hatali {wrong}")

    print("\n6. GORUNTU BUTUNLUGU")
    if not os.path.isdir(IMG_ROOT):
        skip("her kaydin goruntusu var",
             f"goruntu dizini yok ({IMG_ROOT}); --img-root ile belirtilebilir")
    else:
        missing, empty = 0, []
        for v in recs.values():
            for r in v:
                w = r[0]
                p = os.path.join(IMG_ROOT, w.split("-")[0], form_of(w), w + ".png")
                if not os.path.exists(p):
                    missing += 1
                elif os.path.getsize(p) == 0:
                    empty.append(w)
        check("her kaydin goruntusu var", missing == 0, f"eksik {missing}")
        # IAM dagitiminda iki dosya 0 bayt gelir: a01-117-05-02 ve
        # r06-022-03-05. Bunlar egitim sirasinda atlanir (skipped:2). Baska
        # bozuk dosya cikarsa veri kopyasi eksik indirilmis demektir.
        KNOWN_EMPTY = {"a01-117-05-02", "r06-022-03-05"}
        unexpected = sorted(set(empty) - KNOWN_EMPTY)
        check("bozuk (0 bayt) goruntu yalniz bilinen 2 dosya",
              not unexpected,
              f"toplam {len(empty)} bos" +
              (f", BEKLENMEYEN: {', '.join(unexpected[:5])}" if unexpected
               else " (ikisi de bilinen, egitimde atlanir)"))

    ok = all(results)
    print(f"\n{'=' * 60}")
    print(f"  {sum(results)}/{len(results)} kontrol gecti - "
          f"{'HEPSI TEMIZ' if ok else 'SORUN VAR'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
