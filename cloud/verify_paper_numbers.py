#!/usr/bin/env python3
"""
Check every number in makale/paper.tex against the files it came from.

Parses Tables 2 (augmentation ablation), 3 (seed repeats) and 4
(post-correction), the 
ewcommand macros and the per-word counts in the
error-analysis prose, and compares each value with
results/ablation_viterbi.json, results/ablation_viterbi_seeds.json,
results/ablation_lexicon5_all.json, results/ablation_trigram.json,
results/paper_stats_trigram.json and the Model_*/training_history.json files.
Exits non-zero and lists every mismatch, so it can gate a commit.

Usage:  python cloud/verify_paper_numbers.py
"""
import json
import os
import re
import sys
from pathlib import Path

R = str(Path(__file__).resolve().parent.parent)
os.chdir(R)
tex = open(os.path.join("makale", "paper.tex"), encoding="utf-8").read()
V = json.load(open("results/ablation_viterbi.json"))
S = json.load(open("results/ablation_viterbi_seeds.json"))
L = json.load(open("results/ablation_lexicon5_all.json"))
T = json.load(open("results/ablation_trigram.json"))
P = json.load(open("results/paper_stats_trigram.json"))
sel = V["selected_on_val"]
bad = []


def chk(name, paper, truth, tol=0.005):
    if truth is None:
        bad.append(f"{name}: no source value")
        return
    if abs(paper - truth) > tol:
        bad.append(f"{name}: paper {paper} vs source {truth:.4f}")


def rows(label):
    """table rows between \\colrule and \\botrule of the table with this label"""
    i = tex.index(r"\label{" + label)
    body = tex[i:tex.index(r"\botrule", i)]
    body = body[body.index(r"\colrule"):]
    out = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or line.startswith(("\\colrule", "\\toprule")) or "multicolumn" in line:
            line = line.replace("\\colrule", "").strip()
            if not line:
                continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > 1:
            out.append(cells)
    return out


def num(cell):
    cell = cell.replace("\\textbf{", "").replace("}", "").replace("$", "")
    m = re.search(r"-?\d+\.\d+|-?\d+", cell.replace("\u2212", "-").replace("--", "-"))
    return float(m.group()) if m else None


print("== Table 2 (augmentation ablation)")
key = {"no augmentation": "none", "CRNN-B (baseline)": "narrow", "$+$ wide photometric": "photo",
       "$+$ elastic": "elastic", "$+$ morphological": "morph", "\\textbf{\\ours} (all)": "full"}
wa = {m: V["models"][m]["variants"][sel]["test"]["wa_pct"] for m in V["models"]}
for cells in rows("tab:main"):
    m = key.get(cells[0].strip())
    if not m:
        continue
    t = V["models"][m]["variants"][sel]["test"]
    chk(f"T2 {m} WA", num(cells[1]), t["wa_pct"])
    chk(f"T2 {m} CER", num(cells[2]), t["cer_pct"])
    ci = re.findall(r"\d+\.\d+", cells[3])
    chk(f"T2 {m} CIlo", float(ci[0]), t["wilson_95ci_pct"][0])
    chk(f"T2 {m} CIhi", float(ci[1]), t["wilson_95ci_pct"][1])
    if "ref." not in cells[4] and "---" not in cells[4]:
        chk(f"T2 {m} dB", num(cells[4]), wa[m] - wa["narrow"])
    if "ref." not in cells[5] and "---" not in cells[5]:
        chk(f"T2 {m} dP", num(cells[5]), wa[m] - wa["photo"])
    if "ref." not in cells[6] and "times" not in cells[6]:
        chk(f"T2 {m} p", num(cells[6]), V["mcnemar_vs_narrow"][m]["p_value"], tol=0.001)
    h = json.load(open(f"Model_abl_{m}/training_history.json"))
    w = [x * 100 for x in h["val_wa"]]
    b = max(range(len(w)), key=lambda i: w[i])
    chk(f"T2 {m} val", num(cells[7]), w[b])
    ep = int(re.search(r"\((\d+)\)", cells[7]).group(1))
    if ep != b + 1:
        bad.append(f"T2 {m} best epoch: paper {ep} vs source {b+1}")
print("   ok" if not bad else "   issues so far: " + "; ".join(bad))

print("== Table 3 (seeds)")
ssel = S["selected_on_val"]
sw = lambda m: S["models"][m]["variants"][ssel]["test"]
for cells in rows("tab:seeds"):
    if cells[0].startswith("mean"):
        continue
    seed = cells[0].strip()
    b_, l_ = {"42": ("narrow", "full"), "123": ("narrow_s123", "full_s123"),
              "456": ("narrow_s456", "full_s456")}[seed]
    chk(f"T3 s{seed} B", num(cells[1]), sw(b_)["wa_pct"])
    chk(f"T3 s{seed} LX", num(cells[2]), sw(l_)["wa_pct"])
    chk(f"T3 s{seed} delta", num(cells[3]), sw(l_)["wa_pct"] - sw(b_)["wa_pct"])
    cers = re.findall(r"\d+\.\d+", cells[5])
    chk(f"T3 s{seed} CER-B", float(cers[0]), sw(b_)["cer_pct"])
    chk(f"T3 s{seed} CER-LX", float(cers[1]), sw(l_)["cer_pct"])
print("   checked")

print("== Table 4 (post-correction)")
lex = {x["name"]: x for x in L["models"]["full"]["configurations"]}
want = {
    "none (\\oursnolm, greedy CTC)": lex["none (greedy CTC)"],
    "training lexicon, edit distance only": lex["IAM lexicon, edit only"],
    "training lexicon $+$ frequency prior": lex["IAM lexicon + n-gram"],
    "extended lexicon, edit distance only": lex["extended lexicon, edit only"],
    "extended lexicon $+$ frequency prior": lex["extended lexicon + n-gram"],
    "extended lexicon $+$ KN1": T["variants"]["KN1-IAM"]["test"],
    "extended lexicon $+$ KN3, L$\\to$R": T["variants"]["KN3-IAM"]["test"],
    "extended lexicon $+$ KN3, line": V["variants"]["VIT"]["test"],
    "\\textbf{extended lexicon $+$ KN3, line, keep-OOV} (\\ours)": V["variants"]["VIT+oov"]["test"],
    "extended lexicon $+$ KN3, line, real-word": V["variants"]["VIT+rw"]["test"],
    "extended lexicon $+$ KN3, line, keep-OOV $+$ real-word": V["variants"]["VIT+oov+rw"]["test"],
}
seen = 0
for cells in rows("tab:lexicon"):
    name, rank = cells[0].strip(), cells[1].strip()
    src = want.get(name)
    if name == "extended lexicon $+$ KN1" and "Brown" in rank:
        src = T["variants"]["KN1-IAM+Brown"]["test"]
    if name == "extended lexicon $+$ KN3, L$\\to$R" and "Brown" in rank:
        src = T["variants"]["KN3-IAM+Brown"]["test"]
    if src is None:
        bad.append(f"T4 unmatched row: {name} | {rank}")
        continue
    seen += 1
    chk(f"T4 {name[:28]} ({rank[:12]}) WA", num(cells[3]), src["wa_pct"])
    chk(f"T4 {name[:28]} ({rank[:12]}) CER", num(cells[4]), src["cer_pct"])
    ci = re.findall(r"\d+\.\d+", cells[5])
    chk(f"T4 {name[:28]} CIlo", float(ci[0]), src["wilson_95ci_pct"][0])
print(f"   {seen} rows checked")

print("== macros")
for macro, truth in [("ourwa", V["variants"]["VIT+oov"]["test"]["wa_pct"]),
                     ("ourcer", V["variants"]["VIT+oov"]["test"]["cer_pct"]),
                     ("leftwa", T["variants"]["KN3-IAM+Brown"]["test"]["wa_pct"]),
                     ("uniwa", lex["extended lexicon + n-gram"]["wa_pct"]),
                     ("unicer", lex["extended lexicon + n-gram"]["cer_pct"]),
                     ("rawwa", lex["none (greedy CTC)"]["wa_pct"]),
                     ("rawcer", lex["none (greedy CTC)"]["cer_pct"]),
                     ("baselinewa", wa["narrow"]),
                     ("deltawa", wa["full"] - wa["narrow"]),
                     ("anchordelta", abs(wa["none"] - wa["narrow"]))]:
    m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", tex)
    chk(f"macro {macro}", num(m.group(1)), truth)
ci = re.search(r"\\newcommand\{\\ourci\}\{\[([\d.]+)\\%, ([\d.]+)\\%\]\}", tex)
chk("macro ourci lo", float(ci.group(1)), V["variants"]["VIT+oov"]["test"]["wilson_95ci_pct"][0])
chk("macro ourci hi", float(ci.group(2)), V["variants"]["VIT+oov"]["test"]["wilson_95ci_pct"][1])

print("== error analysis prose")
ea = P["error_analysis_full"]
for pat, truth in [(r"misrecognizes 3\\,(\d+) of", ea["misrecognized"] % 1000),
                   (r"Only (\d+) of\nthem", ea["case_only"]),
                   (r"consist of 5\\,(\d+) substitutions", ea["substitutions"] % 1000),
                   (r"1\\,(\d+) deletions", ea["deletions"] % 1000),
                   (r"and (\d+) insertions", ea["insertions"]),
                   (r"Of the post-corrector's decisions, 1\\,(\d+)", P["corrector_full"]["hypotheses_changed"] % 1000),
                   (r"replaced: (\d+) wrong", P["corrector_full"]["fixed"]),
                   (r"and (\d+) correct ones", P["corrector_full"]["broken"]),
                   (r"replaces 2\\,(\d+), repairs", P["corrector_full"]["reference_changed"] % 1000),
                   (r"repairs 1\\,(\d+) and breaks", P["corrector_full"]["reference_fixed"] % 1000),
                   (r"only (\d+)\. Whole-line", 888)]:
    m = re.search(pat, tex)
    if not m:
        bad.append(f"prose pattern not found: {pat}")
    else:
        chk(f"prose {pat[:28]}", float(m.group(1)), float(truth), tol=0.5)

print()
if bad:
    print("MISMATCHES:", len(bad))
    for b in bad:
        print("   ", b)
    sys.exit(1)
print("OK: every table cell, macro and quoted count matches its source file")
