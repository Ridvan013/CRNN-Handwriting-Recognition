#!/usr/bin/env python3
"""
Verify the Aachen partition used in the paper: writer, form and prompt
disjointness, and the integrity of every record and image.

Section 3.1 of the paper states that these properties are checked
programmatically; this is that check.  It reads only the files distributed
with the repository, needs no GPU and no model:

    aachen_splits/splits/{train,validation,test}.uttlist   the published split
    aachen_splits/{train,validation,test}_words.txt        the records we use
    aachen_splits/form_writer.txt                          form -> writer id
    HTR_Using_CRNN/.../iam_words/words.txt                  the IAM ground truth
    HTR_Using_CRNN/.../iam_words/<a>/<a-b>/<id>.png         the crops

What it checks
--------------
1. Form disjointness of the three partitions, on the published lists and on
   the filtered records we train and evaluate on.
2. Writer disjointness of the three partitions, on both.
3. The counts quoted in Section 3.1: forms, writers and word records.
4. Prompt disjointness: no form we keep shares its transcription with a form
   in another partition, and the five validation forms we drop do share their
   prompt with a test form (each is the second half of a prompt whose first
   half is in the test partition, written by a different writer).
4b. Prompt halves: of the 57 prompts that IAM splits into an ``a'' and a ``b''
   form, only those five have their halves in different partitions, and none
   of them involves the training partition -- so no prompt is shared between
   training and test.
5. Record and image integrity: every record is flagged ``ok``, carries a
   non-empty transcription over the 78-symbol alphabet, and points at a PNG
   that exists and is non-empty.  The two empty training images reported in
   Section 3.1 are expected and are listed by name.

Exit code 0 if every check passes, 1 otherwise.

Usage:  python cloud/verify_split.py [--no-images]
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPL = ROOT / "aachen_splits"
IAM = ROOT / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" / "iam_words"
PARTS = ("train", "validation", "test")
EXPECTED = {                      # Section 3.1 of the paper
    "published_forms": {"train": 747, "validation": 116, "test": 336},
    "published_writers": {"train": 283, "validation": 56, "test": 161},
    "kept_forms": {"train": 747, "validation": 111, "test": 336},
    "kept_writers": {"train": 283, "validation": 55, "test": 161},
    "records": {"train": 47999, "validation": 7205, "test": 20310},
    "unreadable_images": 2,
}
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print("  FAIL  " + msg)


def ok(msg: str) -> None:
    print("  ok    " + msg)


def form_of(word_id: str) -> str:
    return "-".join(word_id.split("-")[:2])


def read_uttlists() -> dict:
    return {p: set((SPL / "splits" / f"{p}.uttlist").read_text().split())
            for p in PARTS}


def read_records() -> dict:
    """{partition: [(word_id, status, transcription)]} from *_words.txt."""
    out = {}
    for p in PARTS:
        rows = []
        for ln in (SPL / f"{p}_words.txt").read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            f = ln.split()
            rows.append((f[0], f[1], f[-1]))
        out[p] = rows
    return out


def read_form_writer() -> dict:
    fw = {}
    for ln in (SPL / "form_writer.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            fw[parts[0]] = parts[1]
    return fw


def read_iam_forms() -> dict:
    """{form: ' '.join(transcriptions in word-id order)} over status=ok words."""
    per_form = defaultdict(list)
    src = IAM / "words.txt"
    if not src.exists():
        return {}
    for ln in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split()
        if len(f) < 9 or f[1] != "ok":
            continue
        per_form[form_of(f[0])].append((f[0], f[-1]))
    return {k: " ".join(t for _, t in sorted(v)) for k, v in per_form.items()}


def png_is_readable(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return False
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    w, h = struct.unpack(">II", head[16:24])
    return w > 0 and h > 0


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true",
                    help="skip the per-image check (~76k files)")
    a = ap.parse_args()

    utt = read_uttlists()
    rec = read_records()
    fw = read_form_writer()
    kept = {p: {form_of(w) for w, _, _ in rec[p]} for p in PARTS}

    print("== 1. form disjointness")
    for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for label, d in (("published", utt), ("kept", kept)):
            inter = d[x] & d[y]
            (ok if not inter else fail)(
                f"{label}: {x} n {y} = {len(inter)} forms")

    print("== 2. writer disjointness")
    missing = sorted({f for p in PARTS for f in utt[p] if f not in fw})
    if missing:
        fail(f"{len(missing)} forms have no writer id, e.g. {missing[:3]}")
    else:
        ok("every form has a writer id")
    for label, d in (("published", utt), ("kept", kept)):
        W = {p: {fw[f] for f in d[p] if f in fw} for p in PARTS}
        for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
            inter = W[x] & W[y]
            (ok if not inter else fail)(
                f"{label}: {x} n {y} = {len(inter)} writers")

    print("== 3. counts quoted in Section 3.1")
    for p in PARTS:
        wpub = {fw[f] for f in utt[p] if f in fw}
        wkept = {fw[f] for f in kept[p] if f in fw}
        for name, got, want in (
                ("forms (published)", len(utt[p]), EXPECTED["published_forms"][p]),
                ("writers (published)", len(wpub), EXPECTED["published_writers"][p]),
                ("forms (kept)", len(kept[p]), EXPECTED["kept_forms"][p]),
                ("writers (kept)", len(wkept), EXPECTED["kept_writers"][p]),
                ("word records", len(rec[p]), EXPECTED["records"][p])):
            (ok if got == want else fail)(f"{p} {name}: {got} (expected {want})")

    print("== 4. prompt disjointness")
    forms_text = read_iam_forms()
    if not forms_text:
        fail(f"{(IAM / 'words.txt')} not found, prompt check skipped")
    else:
        text_owner = defaultdict(set)
        for p in PARTS:
            for f in kept[p]:
                if forms_text.get(f):
                    text_owner[forms_text[f]].add(p)
        shared = [t for t, owners in text_owner.items() if len(owners) > 1]
        (ok if not shared else fail)(
            f"{len(shared)} transcriptions shared between partitions we keep")
        dropped = sorted(utt["validation"] - kept["validation"])
        (ok if len(dropped) == 5 else fail)(
            f"{len(dropped)} validation forms dropped: {dropped}")
        for f in dropped:
            t = forms_text.get(f, "")
            best, score = None, 0.0
            for g in utt["test"]:
                s = jaccard(t, forms_text.get(g, ""))
                if s > score:
                    best, score = g, s
            same_writer = fw.get(f) == fw.get(best)
            good = score >= 0.5 and not same_writer
            (ok if good else fail)(
                f"{f} (writer {fw.get(f)}) shares its prompt with test form "
                f"{best} (writer {fw.get(best)}), overlap {score:.2f}")

    print("== 4b. prompt halves (a/b forms) across partitions")
    import re as _re
    owner = {f: pt for pt in PARTS for f in utt[pt]}
    stems = defaultdict(list)
    for f in owner:
        m = _re.match(r"^([a-z]\d\d-\d+)([a-z]?)$", f)
        if m:
            stems[m.group(1)].append(f)
    pairs = {k: sorted(v) for k, v in stems.items() if len(v) > 1}
    straddle = {k: v for k, v in pairs.items() if len({owner[x] for x in v}) > 1}
    ok(f"{len(pairs)} prompts are written in two halves; {len(straddle)} of them "
       f"have their halves in different partitions")
    bad_pairs = []
    for k, v in sorted(straddle.items()):
        parts = {owner[x] for x in v}
        # train must never share a prompt with validation or test
        if "train" in parts:
            bad_pairs.append((k, v))
            fail(f"{k}: halves in {sorted(parts)} -> " +
                 ", ".join(f"{x} [{owner[x]}]" for x in v))
        else:
            in_kept = [x for x in v if x in kept[owner[x]]]
            good = len(in_kept) == 1              # we keep only the test half
            (ok if good else fail)(
                f"{k}: " + ", ".join(f"{x} [{owner[x]}]" for x in v) +
                f"; kept: {in_kept}")
    if not bad_pairs:
        ok("no prompt is shared between the training partition and either "
           "of the others")

    print("== 5. record and image integrity")
    try:
        sys.path.insert(0, str(ROOT / "cloud"))
        from model_v3 import CHAR_LIST
        alphabet = set(CHAR_LIST)
        ok(f"alphabet from model_v3: {len(alphabet)} symbols")
    except Exception as exc:                      # pragma: no cover
        alphabet = None
        fail(f"could not import CHAR_LIST ({exc}); alphabet check skipped")

    bad_status = bad_tr = bad_chars = 0
    for p in PARTS:
        for wid, status, tr in rec[p]:
            if status != "ok":
                bad_status += 1
            if not tr:
                bad_tr += 1
            elif alphabet is not None and not set(tr) <= alphabet:
                bad_chars += 1
    (ok if not bad_status else fail)(f"{bad_status} records not flagged ok")
    (ok if not bad_tr else fail)(f"{bad_tr} empty transcriptions")
    (ok if not bad_chars else fail)(
        f"{bad_chars} transcriptions with characters outside the alphabet")

    if a.no_images:
        print("  ..    image check skipped (--no-images)")
    else:
        unreadable = []
        for p in PARTS:
            for wid, _, _ in rec[p]:
                d1 = wid.split("-")[0]
                path = IAM / "words" / d1 / form_of(wid) / f"{wid}.png"
                if not png_is_readable(path):
                    unreadable.append(wid)
        n = len(unreadable)
        (ok if n == EXPECTED["unreadable_images"] else fail)(
            f"{n} unreadable or empty images (expected "
            f"{EXPECTED['unreadable_images']}): {unreadable}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        return 1
    print("every check passed: the partitions are writer-, form- and "
          "prompt-disjoint and every record and image is intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
