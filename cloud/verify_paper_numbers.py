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

print()
if bad:
    print("MISMATCHES:", len(bad))
    for b in bad:
        print("   ", b)
    sys.exit(1)
print("OK: every table cell, macro and quoted count matches its source file")
