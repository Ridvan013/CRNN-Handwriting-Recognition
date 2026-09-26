#!/usr/bin/env python3
"""
Verify the Aachen word-level partition used in the paper.

Checks, independently of the training code, every structural property the
paper claims in Section 3.1: form disjointness, WRITER disjointness, exact
agreement with the official uttlists, prompt disjointness, how the forms that
share a prompt are distributed over the partitions, the published and the used
writer counts, and the integrity of every record and image.

Run:
    python verify_aachen_splits.py [--img-root PATH]

The image directory can also be given through the IAM_ROOT environment
variable; on Kaggle the crops live under /kaggle/input.

Exit code 0 = every check passed, 1 = at least one check failed.
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(ROOT, "aachen_splits")
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
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, detail=""):
    """A check the environment cannot perform; not counted as a failure."""
    print(f"  [SKIP] {name}" + (f"  {detail}" if detail else ""))


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

    print("\nPARTITION SIZES")
    for k in ("train", "val", "test"):
        print(f"  {k:<6} {len(forms[k]):>4} forms  {len(recs[k]):>7,} words")

    print("\n1. FORM DISJOINTNESS")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = forms[a] & forms[b]
        check(f"{a} vs {b}", not ov, f"shared {len(ov)}")

    print("\n2. AGREEMENT WITH THE OFFICIAL LISTS")
    check("test is the COMPLETE official list", forms["test"] == utt["test"],
          f"{len(forms['test'])}/{len(utt['test'])}")
    check("train is the COMPLETE official list", forms["train"] == utt["train"],
          f"{len(forms['train'])}/{len(utt['train'])}")
    check("val is a SUBSET of the official list", forms["val"] <= utt["val"],
          f"{len(forms['val'])}/{len(utt['val'])} "
          f"({len(utt['val'] - forms['val'])} forms dropped for prompt overlap)")

    print("\n3. WRITER DISJOINTNESS")
    fw_path = os.path.join(SPLIT_DIR, "form_writer.txt")
    if not os.path.exists(fw_path):
        check("form_writer.txt present", False,
              "file missing, writer check NOT PERFORMED")
    else:
        f2w = {}
        with open(fw_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    a, b = line.split()
                    f2w[a] = b
        unknown = [f for v in forms.values() for f in v if f not in f2w]
        check("every form has a known writer", not unknown, f"missing {len(unknown)}")
        writers = {k: {f2w[f] for f in v if f in f2w} for k, v in forms.items()}
        for k in ("train", "val", "test"):
            print(f"         {k:<6} {len(writers[k]):>4} writers")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = writers[a] & writers[b]
            check(f"{a} vs {b}", not ov,
                  f"shared {len(ov)}" + (f" -> {sorted(ov)[:5]}" if ov else ""))

    print("\n4. PROMPT DISJOINTNESS")
    bases = {k: {text_base(f) for f in v} for k, v in forms.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = bases[a] & bases[b]
        check(f"{a} vs {b}", not ov, f"shared {len(ov)}")

    print("\n4b. FORMS THAT SHARE A PROMPT, AND THE WRITER COUNTS")
    # IAM writes some of its prompt texts out in more than one form: the form
    # id keeps the prompt number and adds a letter (a/b, u/x, a..m).  Forms
    # that share the numeric stem therefore share the prompt.  Section 3.1
    # states that 57 prompts are written out by more than one form, that
    # exactly five of those groups straddle partitions, that none of them
    # involves the training partition, and that the published validation list
    # has 56 writers against the 55 that remain after the five are dropped.
    owner = {f: k for k in ("train", "val", "test") for f in utt[k]}
    stems = collections.defaultdict(list)
    for f in owner:
        m = re.match(r"^([a-z]\d\d-\d+)([a-z]*)$", f)
        if m:
            stems[m.group(1)].append(f)
    groups = {k: sorted(v) for k, v in stems.items() if len(v) > 1}
    sizes = sorted(len(v) for v in groups.values())
    check("57 prompts are written out by more than one form", len(groups) == 57,
          f"found {len(groups)}, group size {sizes[0]}-{sizes[-1]}")
    straddle = {k: v for k, v in groups.items()
                if len({owner[x] for x in v}) > 1}
    check("exactly 5 of those groups straddle partitions", len(straddle) == 5,
          f"found {len(straddle)}: {', '.join(sorted(straddle))}")
    with_train = [k for k, v in straddle.items()
                  if "train" in {owner[x] for x in v}]
    check("none of them involves the training partition", not with_train,
          f"overlapping with train {len(with_train)}" +
          (f" -> {with_train}" if with_train else ""))
    kept_one_test = [k for k, v in straddle.items()
                     if [x for x in v if x in forms[owner[x]]] ==
                        [x for x in v if x in utt["test"]]]
    check("in those groups only the test form is kept",
          len(kept_one_test) == len(straddle),
          f"{len(kept_one_test)}/{len(straddle)}")
    if os.path.exists(fw_path):
        pub = len({f2w[f] for f in utt["val"] if f in f2w})
        kept = len({f2w[f] for f in forms["val"] if f in f2w})
        check("the published validation list has 56 writers", pub == 56,
              f"found {pub}")
        check("55 writers remain once the five forms are dropped", kept == 55,
              f"found {kept}")
    else:
        skip("published writer counts", "form_writer.txt missing")

    print("\n5. RECORD INTEGRITY")
    bad_status = sum(1 for v in recs.values() for r in v if r[1] != "ok")
    check("every record is flagged ok", bad_status == 0, f"violations {bad_status}")
    ids = [r[0] for v in recs.values() for r in v]
    check("no duplicated word_id", len(ids) == len(set(ids)),
          f"duplicates {len(ids) - len(set(ids))}")
    junk = sum(1 for v in recs.values() for r in v if not re.match(r"^[a-z]\d+-", r[0]))
    check("no record from outside IAM", junk == 0, f"found {junk}")
    wrong = sum(1 for k, v in recs.items() for r in v if form_of(r[0]) not in utt[k])
    check("every record sits in the right partition", wrong == 0, f"misplaced {wrong}")

    print("\n6. IMAGE INTEGRITY")
    if not os.path.isdir(IMG_ROOT):
        skip("every record has an image",
             f"image directory missing ({IMG_ROOT}); pass --img-root")
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
        check("every record has an image", missing == 0, f"missing {missing}")
        # Two files arrive empty in the IAM distribution: a01-117-05-02 and
        # r06-022-03-05.  Training skips them (skipped:2).  Any further empty
        # file means the local copy of the data was downloaded incompletely.
        KNOWN_EMPTY = {"a01-117-05-02", "r06-022-03-05"}
        unexpected = sorted(set(empty) - KNOWN_EMPTY)
        check("the only empty (0-byte) images are the 2 known ones",
              not unexpected,
              f"{len(empty)} empty in total" +
              (f", UNEXPECTED: {', '.join(unexpected[:5])}" if unexpected
               else " (both known, skipped during training)"))

    ok = all(results)
    print(f"\n{'=' * 60}")
    print(f"  {sum(results)}/{len(results)} checks passed - "
          f"{'ALL CLEAN' if ok else 'PROBLEM FOUND'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
