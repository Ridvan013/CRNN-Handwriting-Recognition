#!/usr/bin/env python3
"""
Check every number in makale/paper.tex against the file it came from.

Parses the four data tables, the \\newcommand macros and the counts quoted in
the prose, and compares each value with

    results/ablation_final.json          Table 2 (augmentation ablation)
    results/ablation_final_seeds.json    Table 3 (seed repeats)
    results/ablation_lexicon_source.json Table 4 (lexicon x corrector)
    results/ablation_wbs.json            Table 5 (word beam search)
    results/paper_stats_final.json       per-word counts in Sections 5.1/5.4
    Model_*/training_history.json        the best-val-WA column of Table 2

Exits non-zero and lists every mismatch, so it can gate a commit.

Usage:  python cloud/verify_paper_numbers.py
"""
import csv
import json
import os
import re
import sys
from math import comb
from pathlib import Path

R = str(Path(__file__).resolve().parent.parent)
os.chdir(R)
BS = chr(92)
tex = open(os.path.join("makale", "paper.tex"), encoding="utf-8").read()
FIN = json.load(open("results/ablation_final.json", encoding="utf-8"))
SEED = json.load(open("results/ablation_final_seeds.json", encoding="utf-8"))
LS = json.load(open("results/ablation_lexicon_source.json", encoding="utf-8"))
WB = json.load(open("results/ablation_wbs.json", encoding="utf-8"))
PS = json.load(open("results/paper_stats_final.json", encoding="utf-8"))
bad = []


def chk(name, paper, truth, tol=0.005):
    if paper is None:
        bad.append(f"{name}: not found in the paper")
        return
    if abs(paper - truth) > tol:
        bad.append(f"{name}: paper {paper} vs source {truth:.4f}")


def rows(label):
    i = tex.index("\\label{" + label)
    body = tex[i:tex.index("\\botrule", i)]
    body = body[body.index("\\colrule"):]
    out = []
    for line in body.split("\\\\"):
        line = line.replace("\\colrule", "").strip()
        if not line or "multicolumn" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > 1:
            out.append(cells)
    return out


def num(cell):
    cell = cell.replace("\\textbf{", "").replace("}", "").replace("$", "")
    cell = cell.replace("\u2212", "-").replace("--", "-")
    cell = cell.replace(chr(92) + ",", "")
    m = re.search(r"-?\d+\.\d+|-?\d+", cell)
    return float(m.group()) if m else None


def pval(cell):
    r"""A p value written as 0.049 or as $1\!\times\!10^{-22}$."""
    c = cell.replace(BS, "").replace("!", "").replace("$", "").replace(" ", "")
    m = re.search("([0-9.]+)times10[\\^]+{(-?[0-9]+)}", c)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return num(cell)


def chk_p(name, paper, truth):
    """p values are compared on a relative scale: the paper rounds them."""
    if paper is None:
        bad.append(f"{name}: not found in the paper")
        return
    if truth == 0 or not 0.66 <= paper / truth <= 1.5:
        bad.append(f"{name}: paper {paper:g} vs source {truth:g}")


def pair(cell):
    """'83.80 (7.96)' -> (83.80, 7.96)"""
    m = re.findall(r"\d+\.\d+", cell.replace("\\textbf{", "").replace("}", ""))
    return (float(m[0]), float(m[1])) if len(m) >= 2 else (None, None)


# ---------------------------------------------------------------- Table 2
print("== Table 2 (augmentation ablation)")
KEY = {"no augmentation": "none", "CRNN-B (baseline)": "narrow",
       "$+$ wide photometric": "photo", "$+$ elastic": "elastic",
       "$+$ morphological": "morph", "\\textbf{\\ours} (all)": "full"}
wa = {m: FIN["models"][m]["test"]["final"]["wa_pct"] for m in FIN["models"]}
seen = 0
for cells in rows("tab:main"):
    m = KEY.get(cells[0].strip())
    if not m:
        continue
    seen += 1
    t = FIN["models"][m]["test"]["final"]
    chk(f"T2 {m} WA", num(cells[1]), t["wa_pct"])
    chk(f"T2 {m} CER", num(cells[2]), t["cer_pct"])
    ci = re.findall(r"\d+\.\d+", cells[3])
    chk(f"T2 {m} CI", float(ci[0]), t["wilson_95ci_pct"][0])
    chk(f"T2 {m} CI", float(ci[1]), t["wilson_95ci_pct"][1])
    if "ref." not in cells[4] and "---" not in cells[4]:
        chk(f"T2 {m} dB", num(cells[4]), wa[m] - wa["narrow"])
    if "ref." not in cells[5] and "---" not in cells[5]:
        chk(f"T2 {m} dP", num(cells[5]), wa[m] - wa["photo"])
    if "ref." not in cells[6]:
        chk_p(f"T2 {m} p", pval(cells[6]), FIN["mcnemar_vs_narrow"][m]["p_value"])
    h = json.load(open(f"Model_abl_{m}/training_history.json", encoding="utf-8"))
    w = [x * 100 for x in h["val_wa"]]
    b = max(range(len(w)), key=lambda i: w[i])
    chk(f"T2 {m} val", num(cells[7]), w[b])
    if int(re.search(r"\((\d+)\)", cells[7]).group(1)) != b + 1:
        bad.append(f"T2 {m} best epoch mismatch")
print(f"   {seen} rows")

# ---------------------------------------------------------------- Table 3
print("== Table 3 (seed repeats)")
sw = {m: SEED["models"][m]["test"]["final"] for m in SEED["models"]}
PAIRS = {"42": ("narrow", "full"), "123": ("narrow_s123", "full_s123"),
         "456": ("narrow_s456", "full_s456")}
seen = 0
for cells in rows("tab:seeds"):
    seed = cells[0].strip()
    if seed not in PAIRS:
        continue
    seen += 1
    b, l = PAIRS[seed]
    chk(f"T3 s{seed} B", num(cells[1]), sw[b]["wa_pct"])
    chk(f"T3 s{seed} LX", num(cells[2]), sw[l]["wa_pct"])
    chk(f"T3 s{seed} delta", num(cells[3]), sw[l]["wa_pct"] - sw[b]["wa_pct"])
    cers = re.findall(r"\d+\.\d+", cells[5])
    chk(f"T3 s{seed} CER-B", float(cers[0]), sw[b]["cer_pct"])
    chk(f"T3 s{seed} CER-LX", float(cers[1]), sw[l]["cer_pct"])
print(f"   {seen} seeds")

print("== Table 3 p column (within-seed McNemar, recomputed)")
PRED = "results/preds_final_seeds/preds_%s.csv"


def correct_by_id(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["word_id"]: int(r["correct"]) for r in csv.DictReader(f)}


def mcnemar(a, b):
    ids = sorted(a)
    x = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    y = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


for cells in rows("tab:seeds"):
    seed = cells[0].strip()
    if seed not in PAIRS:
        continue
    b, l = PAIRS[seed]
    A = correct_by_id(PRED % b)
    B = correct_by_id(PRED % l)
    chk_p(f"T3 s{seed} p", pval(cells[4]), mcnemar(A, B))
print("   3 seeds")

# ---------------------------------------------------------------- Table 4
print("== Table 4 (lexicon x corrector)")
LEXKEY = {"training": "training (7K)", "word list": "extended (239K)",
          "corpus": "corpus vocabulary (57K)"}
CORR = ["edit distance only", "unigram prior", "KN3 left-to-right",
        "KN3 whole-line, keep-OOV"]
seen = 0
for cells in rows("tab:lexicon"):
    key = LEXKEY.get(cells[0].strip())
    if not key:
        continue
    seen += 1
    e = LS["lexicons"][key]["correctors"]
    chk(f"T4 {key} types", num(cells[1]), LS["lexicons"][key]["types"], tol=0.5)
    chk(f"T4 {key} coverage", num(cells[2]), LS["lexicons"][key]["test_coverage_pct"], tol=0.05)
    for i, c in enumerate(CORR):
        w, cer = pair(cells[3 + i])
        chk(f"T4 {key} / {c} WA", w, e[c]["test"]["wa_pct"])
        chk(f"T4 {key} / {c} CER", cer, e[c]["test"]["cer_pct"])
print(f"   {seen} lexicons x {len(CORR)} correctors")

# ---------------------------------------------------------------- Table 5
print("== Table 5 (word beam search)")
WKEY = {("training lexicon", "7"): "WBS Words, 7K",
        ("training lexicon $+$ case", "13"): "WBS Words, 7K, case variants",
        ("word-list lexicon", "239"): "WBS Words, 239K",
        ("word-list lexicon $+$ case", "473"): "WBS Words, 239K, case variants",
        ("corpus lexicon", "57"): "WBS Words, corpus vocab",
        ("corpus lexicon $+$ case", "100"): "WBS Words, corpus vocab, case variants",
        ("corpus text (bigram LM)", "57"): "WBS NGrams, IAM+Brown"}
ours = LS["lexicons"]["corpus vocabulary (57K)"]["correctors"]["KN3 left-to-right"]["test"]
seen = 0
for cells in rows("tab:wbs"):
    dic = cells[1].strip()
    key = next((v for (d, _), v in WKEY.items() if d == dic), None) if "WBS" in cells[0] else None
    if key is None:
        if "post-correction" in cells[0]:
            chk("T5 ours WA", num(cells[3]), ours["wa_pct"])
            chk("T5 ours CER", num(cells[4]), ours["cer_pct"])
            seen += 1
        continue
    seen += 1
    t = WB["configs"][key]["test"]
    chk(f"T5 {key} WA", num(cells[3]), t["wa_pct"])
    chk(f"T5 {key} CER", num(cells[4]), t["cer_pct"])
    de = WB["configs"][key].get("dictionary_entries")
    if de:
        chk(f"T5 {key} types", num(cells[2]), float(de), tol=0.5)
print(f"   {seen} rows")

# ---------------------------------------------------------------- macros
print("== macros")
L = {k: {c: v["test"] for c, v in e["correctors"].items()} for k, e in LS["lexicons"].items()}
CO, WL = "corpus vocabulary (57K)", "extended (239K)"
MACROS = [("ourwa", ours["wa_pct"]), ("ourcer", ours["cer_pct"]),
          ("uniwa", L[CO]["unigram prior"]["wa_pct"]),
          ("unicer", L[CO]["unigram prior"]["cer_pct"]),
          ("listwa", L[WL]["KN3 left-to-right"]["wa_pct"]),
          ("listcer", L[WL]["KN3 left-to-right"]["cer_pct"]),
          ("rawwa", LS["greedy"]["wa_pct"]), ("rawcer", LS["greedy"]["cer_pct"]),
          ("baselinewa", wa["narrow"]), ("deltawa", wa["full"] - wa["narrow"]),
          ("anchordelta", abs(wa["none"] - wa["narrow"]))]
for macro, truth in MACROS:
    m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", tex)
    chk(f"macro {macro}", num(m.group(1)) if m else None, truth)
ci = re.search(r"\\newcommand\{\\ourci\}\{\[([\d.]+)\\%, ([\d.]+)\\%\]\}", tex)
if ci:
    chk("macro ourci lo", float(ci.group(1)), ours["wilson_95ci_pct"][0])
    chk("macro ourci hi", float(ci.group(2)), ours["wilson_95ci_pct"][1])
else:
    bad.append("macro ourci: not found")

# ---------------------------------------------------------------- prose
print("== prose counts")
ea = PS["error_analysis_full"]
cf = PS["corrector_full"]
a5 = PS["augmented_five"]
pw = PS["pairwise_only_correct"]
def after(phrase, nth=1):
    """The nth number following `phrase`; thin-space separators are stripped."""
    i = tex.find(phrase)
    if i < 0:
        return None
    window = tex[i + len(phrase): i + len(phrase) + 240]
    nums = re.findall(r"\d[\d]*(?:\\,\d{3})*(?:\.\d+)?", window)
    vals = [float(n.replace(chr(92) + ",", "")) for n in nums]
    return vals[nth - 1] if len(vals) >= nth else None


PROSE = [
    ("misrecognizes", 1, ea["misrecognized"]),
    ("consist of", 1, ea["substitutions"]),
    ("substitutions,", 1, ea["deletions"]),
    ("deletions and", 1, ea["insertions"]),
    ("that corrector changes", 1, cf["hypotheses_changed"]),
    ("repairs", 1, cf["fixed"]),
    ("and breaks", 1, cf["broken"]),
    ("these five disagree on", 1, a5["disagree"]),
    ("pair,", 1, pw["full|narrow"]["only_narrow"]),
    ("pair,", 2, pw["full|narrow"]["only_full"]),
    ("only by the baseline against", 1, pw["narrow|none"]["only_none"]),
    ("word-list lexicon changed", 1, cf["reference_changed"]),
    ("word-list lexicon changed", 2, cf["reference_fixed"]),
    ("word-list lexicon changed", 3, cf["reference_broken"]),
    ("16.2\\%. Only", 1, ea["case_only"]),
]
for phrase, nth, truth in PROSE:
    chk("prose " + repr(phrase[:32]), after(phrase, nth), float(truth), tol=0.5)

# ------------------------------------------- Section 3.5 corpus and leakage
print("== external corpus and leakage (Section 3.5)")
BL = json.load(open("results/brown_leakage.json", encoding="utf-8"))
po = BL["per_order"]
chk("Brown sentences", after("cite{francis1979brown}", 1),
    float(BL["brown_sentences"]), tol=0.5)
chk("Brown tokens (M)", after("cite{francis1979brown}", 2), BL["brown_tokens"] / 1e6, tol=0.005)
chk("leak 3-gram", after("samples of English are:", 1), po["3"]["pct_in_brown"], tol=0.05)
chk("leak 4-gram", after("on the test lines and", 1), po["4"]["pct_in_brown"], tol=0.05)
chk("leak 5-gram", after("Longer ones are not: of the 5-grams", 1),
    po["5"]["pct_in_brown"], tol=0.005)
chk("leak 6-gram", after("of the 6-grams", 1), po["6"]["pct_in_brown"], tol=0.005)
chk("leak IAM 3-gram", after("IAM training lines gives", 1),
    po["3"]["pct_in_iam_train"], tol=0.05)
chk("leak IAM 5-gram", after("IAM training lines gives", 2),
    po["5"]["pct_in_iam_train"], tol=0.005)
if po["8"]["pct_in_brown"] != 0 or BL["longest_shared_ngram"] != 7:
    bad.append("leakage: the 8-gram/longest claim no longer holds")

# ------------------------------------------- Section 3.5 / 6.2 OOV counts
print("== out-of-lexicon counts (Sections 3.5 and 6.2)")
OO = json.load(open("results/oov_hypotheses.json", encoding="utf-8"))["lexicons"]
co = OO["corpus vocabulary (57K)"]
chk("prose OOV hypotheses", after("On the test set", 1),
    float(co["hypotheses_out_of_lexicon"]), tol=0.5)
chk("prose OOV share", after("On the test set", 2),
    co["hypotheses_out_of_lexicon_pct"], tol=0.05)
chk("prose OOV reachable", after("On the test set", 3),
    float(co["of_those_with_a_candidate"]), tol=0.5)
chk("prose 6.2 OOV share", after("the corrector only ever touches", 1),
    co["hypotheses_out_of_lexicon_pct"], tol=0.05)
chk("prose word-list changed", after("The corrector of the word-list lexicon changed", 1),
    float(OO["extended (239K)"]["of_those_with_a_candidate"]), tol=0.5)

# ------------------------------------- Section 6.2 sums and 6.4 variants
print("== derived figures in Sections 4.4, 6.2 and 6.4")
TR, WL, CO = "training (7K)", "extended (239K)", "corpus vocabulary (57K)"
step = {}
for lx in (WL, CO):
    c = L[lx]
    step[lx] = ((c["unigram prior"]["wa_pct"] - c["edit distance only"]["wa_pct"])
                + (c["KN3 left-to-right"]["wa_pct"] - c["unigram prior"]["wa_pct"]))
chk("6.2 rank+context on corpus", after("contribute together (", 1), step[CO], tol=0.005)
chk("6.2 rank+context on word list", after("contribute together (", 2), step[WL], tol=0.005)
chk("6.2 edit-only gap", after("for edit distance alone it is", 1),
    L[CO]["edit distance only"]["wa_pct"] - L[WL]["edit distance only"]["wa_pct"],
    tol=0.05)
VI = json.load(open("results/ablation_viterbi.json", encoding="utf-8"))
vf = VI["models"]["full"]["variants"]
chk("6.4 real-word variant", after("by a different route (", 1),
    vf["VIT+rw"]["test"]["wa_pct"], tol=0.005)
chk("6.4 selected whole-line", after("by a different route (", 2),
    vf["VIT+oov"]["test"]["wa_pct"], tol=0.005)

import statistics
med, mx, tot = [], [], []
for m in ("none", "narrow", "photo", "elastic", "morph", "full"):
    et = json.load(open(f"Model_abl_{m}/training_history.json",
                        encoding="utf-8"))["epoch_time_s"]
    med.append(statistics.median(et))
    mx.append(max(et))
    tot.append(sum(et) / 3600)
chk("4.4 median epoch lo", after("with seed 42, at a median of", 1), min(med), tol=0.5)
chk("4.4 median epoch hi", after("with seed 42, at a median of", 2), max(med), tol=0.5)
chk("4.4 slowest epoch", after("single epochs up to", 1), max(mx), tol=0.5)
chk("4.4 hours lo", after("one configuration takes", 1), min(tot), tol=0.05)
chk("4.4 hours hi", after("one configuration takes", 2), max(tot), tol=0.05)
chk("4.4 hours total", after("and the six together", 1), sum(tot), tol=0.05)

# ------------------------------- Section 6.4: the per-model corrector ranges
print("== per-model corrector effects (Section 6.4)")
A5 = json.load(open("results/ablation_lexicon5_all.json", encoding="utf-8"))
VI2 = json.load(open("results/ablation_viterbi.json", encoding="utf-8"))
MODES = ("none", "narrow", "photo", "elastic", "morph", "full")
per = {}
for m_ in MODES:
    c_ = {x["name"]: x["wa_pct"] for x in A5["models"][m_]["configurations"]}
    per[m_] = {
        "forced": c_["IAM lexicon, edit only"] - c_["none (greedy CTC)"],
        "prior": c_["extended lexicon + n-gram"] - c_["extended lexicon, edit only"],
        "trigram": (VI2["models"][m_]["test"]["KN3-left"]["wa_pct"]
                    - c_["extended lexicon + n-gram"]),
        "alpha": VI2["models"][m_]["alpha_left"],
    }
aug = [m_ for m_ in MODES if m_ != "none"]
P6 = "costs the five augmented models"
chk("6.4 forced lo (five)", after(P6, 1), min(-per[m_]["forced"] for m_ in aug), tol=0.05)
chk("6.4 forced hi (five)", after(P6, 2), max(-per[m_]["forced"] for m_ in aug), tol=0.05)
chk("6.4 forced anchor", after("zero-augmentation anchor only", 1),
    -per["none"]["forced"], tol=0.05)
P7 = "word-list lexicon adds"
chk("6.4 prior lo (five)", after(P7, 1), min(per[m_]["prior"] for m_ in aug), tol=0.05)
chk("6.4 prior hi (five)", after(P7, 2), max(per[m_]["prior"] for m_ in aug), tol=0.05)
chk("6.4 prior anchor", after("to the five but", 1), per["none"]["prior"], tol=0.05)
P8 = "it adds"
chk("6.4 trigram lo (six)", after(P8, 1), min(per[m_]["trigram"] for m_ in MODES), tol=0.005)
chk("6.4 trigram hi (six)", after(P8, 2), max(per[m_]["trigram"] for m_ in MODES), tol=0.005)
if {per[m_]["alpha"] for m_ in MODES} != {7.0}:
    bad.append("6.4: alpha_left is not 7 for every model")
# best validation WA (training loop: word-list lexicon + unigram prior) minus
# the test WA of that same corrector, per model
gap = []
for m_ in MODES:
    h_ = json.load(open(f"Model_abl_{m_}/training_history.json", encoding="utf-8"))
    c_ = {x["name"]: x["wa_pct"] for x in A5["models"][m_]["configurations"]}
    gap.append(100 * max(h_["val_wa"]) - c_["extended lexicon + n-gram"])
P9 = "exceeds the test accuracy of that same corrector by"
chk("6.4 val-test gap lo", after(P9, 1), min(gap), tol=0.05)
chk("6.4 val-test gap hi", after(P9, 2), max(gap), tol=0.05)

# ------------------------- Section 5.2: the whole-line decoder, split in two
print("== whole-line decoder decomposition (Section 5.2)")
KO = json.load(open("results/ablation_keep_oov.json", encoding="utf-8"))["lexicons"]
tr, co = KO["training (7K)"], KO["corpus vocabulary (57K)"]
chk("5.2 joint decoding, training", after("whatever the lexicon:", 1),
    tr["from_line_decoding_pp"], tol=0.005)
chk("5.2 joint decoding, corpus", after("whatever the lexicon:", 2),
    co["from_line_decoding_pp"], tol=0.005)
chk("5.2 keep-OOV, training", after("is what depends on it", 1),
    tr["from_keep_oov_pp"], tol=0.005)
chk("5.2 keep-OOV, corpus", after("is what depends on it", 3),
    abs(co["from_keep_oov_pp"]), tol=0.005)
chk("5.2 forced whole-line WA", after("without keep-OOV", 1),
    co["whole_line_forced"]["test"]["wa_pct"], tol=0.005)
chk("5.2 forced whole-line CER", after("without keep-OOV", 2),
    co["whole_line_forced"]["test"]["cer_pct"], tol=0.005)
vgrid = co["whole_line_forced"]["val_grid"]
vleft = max(LS["lexicons"]["corpus vocabulary (57K)"]["correctors"]
            ["KN3 left-to-right"]["val_grid"].values())
chk("5.2 validation margin", after("without keep-OOV", 3),
    max(vgrid.values()) - vleft, tol=0.005)

# ------------------------- Section 6.4: the writer-level bootstrap
print("== writer-level bootstrap (Section 6.4)")
WB = json.load(open("results/writer_bootstrap.json", encoding="utf-8"))
chk("6.4 per-writer min", after("ranges from", 1), WB["per_writer_wa_pct"]["min"], tol=0.5)
chk("6.4 per-writer max", after("ranges from", 2), WB["per_writer_wa_pct"]["max"], tol=0.5)
chk("6.4 per-writer SD", after("ranges from", 3), WB["per_writer_wa_pct"]["sd"], tol=0.05)
chk("6.4 wilson half-width", after("instead of words widens", 1),
    WB["wilson"]["half_width"], tol=0.005)
chk("6.4 writer half-width", after("instead of words widens", 2),
    WB["bootstrap_writer"]["half_width"], tol=0.02)
chk("6.4 form half-width", after("instead of words widens", 3),
    WB["bootstrap_form"]["half_width"], tol=0.02)
if abs(WB["bootstrap_word"]["half_width"] - WB["wilson"]["half_width"]) > 0.02:
    bad.append("6.4: the word-level bootstrap no longer reproduces Wilson")
PB = json.load(open("results/writer_bootstrap_paired.json", encoding="utf-8"))
chk("6.4 paired, word", after("difference gives", 1),
    PB["bootstrap_word"]["half_width"], tol=0.02)
chk("6.4 paired, writer", after("difference gives", 2),
    PB["bootstrap_writer"]["half_width"], tol=0.02)

# ---------------------------------- Section 3.1: the split, from its files
print("== the partition (Section 3.1)")
SPL = os.path.join(R, "aachen_splits")


def _forms(path):
    out = set()
    for ln in open(path, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            out.add("-".join(ln.split()[0].split("-")[:2]))
    return out


def _records(path):
    return sum(1 for ln in open(path, encoding="utf-8")
               if ln.strip() and not ln.startswith("#"))


fw = {}
for ln in open(os.path.join(SPL, "form_writer.txt"), encoding="utf-8"):
    ln = ln.strip()
    if ln and not ln.startswith("#") and len(ln.split()) >= 2:
        fw[ln.split()[0]] = ln.split()[1]
pub = {p_: set(open(os.path.join(SPL, "splits", p_ + ".uttlist"),
                   encoding="utf-8").read().split())
       for p_ in ("train", "validation", "test")}
kept = {p_: _forms(os.path.join(SPL, p_ + "_words.txt"))
        for p_ in ("train", "validation", "test")}
recs = {p_: _records(os.path.join(SPL, p_ + "_words.txt"))
        for p_ in ("train", "validation", "test")}
wr = lambda fs: len({fw[f] for f in fs if f in fw})          # noqa: E731

A = "Resources SLR56:"
for i, (name, truth) in enumerate([
        ("forms train", len(pub["train"])), ("forms val", len(pub["validation"])),
        ("forms test", len(pub["test"])), ("writers train", wr(pub["train"])),
        ("writers val", wr(pub["validation"])), ("writers test", wr(pub["test"]))], 1):
    chk(f"3.1 published {name}", after(A, i), float(truth), tol=0.5)
for i, (name, truth) in enumerate([
        ("train words", recs["train"] - 2), ("val words", recs["validation"]),
        ("val forms", len(kept["validation"])),
        ("val writers", wr(kept["validation"]))], 1):
    chk(f"3.1 kept {name}", after("This yields", i), float(truth), tol=0.5)

# ---------------------------------- Section 5.5: the m/n directed counts
print("== directed substitution counts (Section 5.5)")
sys.path.insert(0, os.path.join(R, "cloud"))
from paper_stats_trigram import align_ops                      # noqa: E402
from collections import Counter                                # noqa: E402

subs = Counter()
with open("results/preds_final/preds_full.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        _, _, _, prs = align_ops(row["prediction"], row["ground_truth"])
        subs.update(prs)
tenth = sorted(subs.values(), reverse=True)[9]
chk("5.5 m->n", 37.0, float(subs[("m", "n")]), tol=0.5)
chk("5.5 n->m", 43.0, float(subs[("n", "m")]), tol=0.5)
chk("5.5 tenth entry", 44.0, float(tenth), tol=0.5)
for a, b, tot in (("a", "o", 215), ("r", "s", 139), ("l", "t", 95),
                  ("n", "r", 83), ("m", "n", 80), ("a", "e", 77)):
    chk(f"5.5 {a}<->{b}", float(tot), float(subs[(a, b)] + subs[(b, a)]), tol=0.5)

# ----------------- Section 5.1: the CRNN-B / CRNN-LX pair under four correctors
print("== the endpoint pair under four correctors (Section 5.1)")
LP = json.load(open("results/mcnemar_left_pair.json", encoding="utf-8"))["mcnemar_full_vs_narrow"]
U5 = A5["mcnemar_vs_narrow"]["full"]
VL = VI2["mcnemar_vs_narrow"]["full"]
P51 = "the\nunigram prior on the word-list lexicon gives"
chk("5.1 unigram delta", after(P51, 1), round(U5["delta_wa_pp"], 2))
chk_p("5.1 unigram p", after(P51, 2), U5["p_value"])
chk("5.1 trigram delta", after("the trigram on that lexicon", 1), round(LP["delta_wa_pp"], 2))
chk_p("5.1 trigram p", after("the trigram on that lexicon", 2), LP["p_value"])
chk("5.1 whole-line delta", after("its whole-line\nvariant", 1), round(VL["delta_wa_pp"], 2))
chk_p("5.1 whole-line p", after("its whole-line\nvariant", 2), VL["p_value"])
if not all(d["delta_wa_pp"] > 0 and d["p_value"] >= 0.01 for d in (U5, LP, VL)):
    bad.append("5.1: 'the sign never changes and the threshold is never crossed' fails")

# ---------- Section 3.5: what the corpus lexicon leaves uncovered
OO = json.load(open("results/oov_hypotheses.json", encoding="utf-8"))["corpus_uncovered_tokens"]
P35 = "rare lower-case words ("
chk("3.5 uncovered lower-case %", after(P35, 1), OO["by_kind_pct"]["lower_case"], tol=0.5)
chk("3.5 uncovered capitalized %", after("mostly proper\nnouns (", 1), OO["by_kind_pct"]["capitalized"], tol=0.5)
chk("3.5 uncovered non-letter %", after("digits or apostrophes (", 1), OO["by_kind_pct"]["non_letter"], tol=0.5)
for w_ in ("codex", "papyrus"):
    if w_ not in OO["examples"]["lower_case"]:
        bad.append(f"3.5 example {w_} is not an uncovered lower-case token")

# ---------- Section 5.2: cells that trade WA for CER, and cells that lose both
g_ = LS["greedy"]
cells = [c["test"] for lx in LS["lexicons"].values() for c in lx["correctors"].values()
         if "test" in c]
trade = sum(1 for c in cells if c["wa_pct"] > g_["wa_pct"] and c["cer_pct"] > g_["cer_pct"])
lose = sum(1 for c in cells if c["wa_pct"] < g_["wa_pct"] and c["cer_pct"] > g_["cer_pct"])
if len(cells) != 12:
    bad.append(f"5.2: expected 12 Table 4 cells with a test entry, found {len(cells)}")
if "in three cells of Table~\\ref{tab:lexicon} a configuration buys" not in tex or trade != 3:
    bad.append(f"5.2: WA-for-CER trade cells = {trade}, paper says three")
if "and in three more it loses both" not in tex or lose != 3:
    bad.append(f"5.2: cells losing both = {lose}, paper says three")

# ---------- Section 5.3: why WBS gets right what CRNN-LX gets wrong
print("== the WBS-only words by cause (Section 5.3)")
WO = json.load(open("results/wbs_only_breakdown.json", encoding="utf-8"))
WC = WO["only_wbs_by_cause"]
P53 = "The two fail differently, though. Of the"
chk("5.3 only-WBS total", after(P53, 1), WO["only_wbs_correct"], tol=0.5)
chk("5.3 ranked lower", after(P53, 2), WC["ranked_lower"], tol=0.5)
chk("5.3 beyond bound", after("near neighbours that a single greedy string cannot. In", 1),
    WC["beyond_bound"], tol=0.5)
chk("5.3 in lexicon", after("\\emph{neeemary} for \\emph{necessary}), in", 1),
    WC["in_lexicon"], tol=0.5)
chk("5.3 assembled", after("is left alone, and in", 1), WC["reference_not_in_lexicon"], tol=0.5)
chk("Table 5 caption letter-run words", after("fewer distinct words\n(", 1),
    WO["wbs_letter_run_words_corpus"], tol=0.5)
EX = {k: {(h, r) for h, r, _ in v} for k, v in WO["examples"].items()}
for key, pairs in (("ranked_lower", [("wich", "which"), ("mather", "mother")]),
                   ("beyond_bound", [("therooghty", "thoroughly"), ("neeemary", "necessary")]),
                   ("reference_not_in_lexicon", [("mid-way", "mid-way"), ("forth-", "forth-")])):
    for p_ in pairs:
        if p_ not in EX[key]:
            bad.append(f"5.3 example {p_} is not in category {key}")

# ------------------------------------ Table 6 and the prose built on it
print("== published systems (Table 6, Sections 1, 5.4, 6.3, 7)")
# (WER, CER) on the IAM test set, as printed in the primary source:
LIT = {
    "sueiras2018offline": (23.80, 8.80),   # Sueiras et al. 2018; also Kang 2021 Tab. 7
    "kang2018convolve": (17.45, 6.88),     # Kang et al. 2018; also Kass & Vats Tab. 5
    # Kang et al. 2021, candidate fusion LM, TEST set: Table 8 of arXiv:1912.10308.
    # Not the 5.79 / 15.91 pair of Kass & Vats Tab. 5: 15.91 is Kang's Table 5,
    # a VALIDATION-set ablation.
    "kang2021candidate": (15.11, 5.74),
    "kass2022attentionhtr": (15.40, 6.50),  # AttentionHTR, case-sensitive, Tab. 5
    "mondal2022yolo": (29.21, 9.53),        # Mondal et al. 2022
    "(via~": (22.86, 11.08),         # CNN-RNN row of Rajesh et al. Tab. 2
    # HWRCNet: WA 80.08 and CER 9.89 from its Tab. 3 (normal images). Its
    # Tab. 2 prints WER 19.20, which would be WA 80.80; the paper uses the
    # WA printed directly and says so in footnote d.
    "Rajesh et al.~\\cite{rajesh2022hwrcnet}": (100 - 80.08, 9.89),
}
T6 = rows("tab:priorwork")
for key, (wer, cer) in LIT.items():
    hit = [r for r in T6 if key in r[0]]
    if len(hit) != 1:
        bad.append(f"Table 6: {len(hit)} rows match {key!r}")
        continue
    chk(f"Table 6 {key} WA", num(hit[0][3]), 100 - wer)
    chk(f"Table 6 {key} CER", num(hit[0][4]), cer)
flat = re.sub(r"\s+", " ", tex)


def before(pattern):
    """The number written immediately before `pattern` (a regex)."""
    m = re.search(r"(\d+(?:\.\d+)?)(?:\\,pp)?\s*" + pattern, flat)
    return float(m.group(1)) if m else None


LX, G, NC, WBSWA = 83.80, 78.83, 82.91, 84.13
A = {k: 100 - v[0] for k, v in LIT.items()}
chk("5.4 below candidate fusion", before(r"below the candidate-fusion"),
    round(A["kang2021candidate"] - LX, 2))
chk("5.4 below AttentionHTR", before(r"below AttentionHTR"),
    round(A["kass2022attentionhtr"] - LX, 2))
chk("5.4 above Kang 2018", before(r"\\emph\{above\} the attention model"),
    round(LX - A["kang2018convolve"], 2))
chk("5.4 Sueiras lexicon-free margin", before(r"more accurate than the recognizer of Sueiras"),
    round(G - A["sueiras2018offline"], 1), tol=0.05)
chk("5.4 CER range lo", after("At\ncharacter level the gap remains", 1),
    min(v[1] for k, v in LIT.items() if k.startswith(("kang", "kass"))))
chk("5.4 CER range hi", after("At\ncharacter level the gap remains", 2),
    max(v[1] for k, v in LIT.items() if k.startswith(("kang", "kass"))))
top2 = sorted((A["kang2021candidate"], A["kass2022attentionhtr"]))
for where, phrase in (("1 scope", "ahead\nof our optical model, and "),
                      ("7 conclusion", "system is 1.3\\,pp above the latter and ")):
    chk(f"{where}: below strongest lo", after(phrase, 1), round(top2[0] - LX, 1), tol=0.05)
    chk(f"{where}: below strongest hi", after(phrase, 2), round(top2[1] - LX, 1), tol=0.05)
chk("6.3 HWRCNet below LX", before(r"below \\ours\\ \(2\.83"),
    round(LX - A["Rajesh et al.~\\cite{rajesh2022hwrcnet}"], 2))
chk("6.3 HWRCNet below no-context", before(r"below its no-context"),
    round(NC - A["Rajesh et al.~\\cite{rajesh2022hwrcnet}"], 2))
chk("6.3 HWRCNet above CRNN-G", before(r"above \\oursnolm\. It"),
    round(A["Rajesh et al.~\\cite{rajesh2022hwrcnet}"] - G, 2))
chk("1 scope: Kang 2018 recognizer ahead of CRNN-G", before(r"ahead of our optical model"),
    round(A["kang2018convolve"] - G, 1), tol=0.05)
chk("6.3 CRNN-G below Kang 2018", before(r"less accurate without a lexicon"),
    round(A["kang2018convolve"] - G, 2))
chk("6.3 CRNN-G below Kang 2018 (closing)", before(r"below the second"),
    round(A["kang2018convolve"] - G, 2))
# partition sizes: Kang et al. 2018, Sec. 4.1 of the GCPR paper (ok-filtered
# RWTH Aachen partition): 47,981 / 20,305 / 7,554 words
for n_, v_ in enumerate((47981, 20305, 7554), 1):
    chk(f"5.4 Kang 2018 partition size {n_}", after("state the same partition and filtering and report", n_), v_, tol=0.5)
if not (A["kass2022attentionhtr"] > WBSWA > A["kang2018convolve"]):
    bad.append("5.4: WBS is no longer between AttentionHTR and Kang 2018")

print()
if bad:
    print("MISMATCHES:", len(bad))
    for b in bad:
        print("   ", b)
    sys.exit(1)
print("OK: every table cell, macro and quoted count matches its source file")
