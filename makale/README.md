# Paper — Decomposing Lexicon-Assisted Correction for Word-Level Handwritten Text Recognition on IAM: Lexicon Coverage, Frequency Ranking and Line Context

**Headline:** **CRNN-LX** reaches **83.80% word accuracy** (Wilson 95% CI
[83.29%, 84.30%], **CER 7.96%**) on the **complete** IAM Aachen
writer-disjoint test set (N = 20,310 words, 336 forms, 161 unseen writers).
That is 1.3 pp above Kang et al. 2018 (82.55), 0.29 pp below Kang et al. 2021
(84.09) and 0.80 pp below AttentionHTR (84.60) — with a plain CRNN trained
from scratch, no synthetic pre-training, no transfer and no ensembling.

The corrector: the **corpus lexicon** (57,382 types = the vocabulary of the
IAM training lines + the Brown corpus, 96.4% coverage of the test tokens),
candidates within <=2 edits ranked by an interpolated Kneser-Ney **trigram**
whose context is the system's **own output** for the preceding words of the
line. Selected on validation (`cloud/ablation_final.py`).

Without line context (unigram prior, the figure for genuinely isolated
words): 82.91 / 8.43. Lexicon-free (CRNN-G): 78.83 / 8.60.

## 21 Sept: the lexicon dominates (81.95 -> 83.80)

`cloud/ablation_lexicon_source.py` gives the same corrector three lexicons;
`cloud/ablation_wbs.py` runs word beam search (the authors' implementation)
on the same probabilities. All on the CRNN-LX optical model, N = 20,310:

| Lexicon (coverage) | edit only | + unigram | **+ KN3, left ctx** | + KN3, line, keep-OOV | WBS (Words) |
|---|---:|---:|---:|---:|---:|
| training 7K (84.8%) | 76.42 | 77.04 | 77.56 | 81.32 | 76.02 |
| word list 239K (93.9%) | 79.51 | 80.74 | 81.66 | 81.95 | 81.26 (81.70 +case) |
| **corpus 57K (96.4%)** | 81.72 | 82.91 | **83.80** | 82.77 | **84.13** |

Findings:

1. **Composition beats size.** The corpus lexicon is 4x smaller than the NLTK
   word list and 2.14 pp better under the same corrector, because it holds
   proper nouns, inflections and the right capitalization.
2. **"Small lexicon is harmful" was really "forced replacement is harmful".**
   With the 7K lexicon, forced replacement costs 1.8 pp against lexicon-free
   decoding; letting an out-of-lexicon hypothesis survive turns the same
   lexicon into +2.5 pp.
3. **In-decoding vs post-hoc is nearly a wash.** At the matched corpus
   lexicon WBS is 0.33 pp above us in WA (p = 0.014, above our 0.01
   threshold) and 0.27 pp worse in CER. WBS's own bigram LM adds nothing
   (84.13 -> 84.07).
4. Table 2 and Table 3 were re-scored with the final corrector; the
   augmentation conclusion is unchanged (CRNN-LX - CRNN-B = +0.39 pp,
   p = 0.050; three seeds: +0.39 / -0.45 / +0.66, mean +0.20 +- 0.58).

Sources: `results/ablation_final.json`, `results/ablation_final_seeds.json`,
`results/ablation_lexicon_source.json`, `results/ablation_wbs.json`,
`results/paper_stats_final.json`; per-word dumps in `results/preds_final*/`
and `results/preds_wbs*/`. `python cloud/verify_paper_numbers.py` checks every
number in the paper against these files.

## 16 Sept: whole-line (two-sided) decoding

`LineViterbiCorrector` in `cloud/kn_trigram.py`: exact second-order Viterbi
over the candidate columns of a line, objective Σ[log P_KN(w_i | w_{i-2},
w_{i-1}) − α d_i]. Verified against exhaustive search on 400 random layouts
(`cloud/viterbi_selftest.py`). Options and α (grid 1–30) selected on
validation (`cloud/ablation_viterbi.py` → `results/ablation_viterbi.json`):

| Decoder (CRNN-LX, 239K lexicon, KN3 IAM+Brown) | α | val WA | test WA | CER | vs left-to-right |
|---|---:|---:|---:|---:|---|
| left-to-right | 7 | 85.50 | 81.66 | 8.58 | ref. |
| whole line | 10 | 85.66 | 81.77 | 8.54 | +56/−34, p=0.026 |
| **whole line + keep-OOV** (selected on val) | 2 | **85.98** | **81.95** | 8.54 | +394/−335, p=0.032 |
| whole line + real-word | 10 | 85.69 | 81.94 | 8.51 | +141/−86, p=0.0003 (not selected) |
| whole line + keep-OOV + real-word | 10 | 84.50 | 79.71 | 8.42 | collapses |

keep-OOV = an out-of-lexicon hypothesis may stay as a distance-0 candidate
(floor probability); real-word = in-lexicon words may be replaced by
distance-1 neighbours. The selected decoder gains +0.32…+0.44 pp on the five
augmented models (p = 0.002–0.034) and +0.08 on the anchor. It wins by being
conservative: 1,925 replacements, 815 repaired / 181 broken, against
2,939 / 1,076 / 501 for left-to-right.

**Table 2 with this corrector:** CRNN-LX − CRNN-B = +0.32 pp, p = 0.132
(unigram 0.37 / 0.063; left-to-right trigram 0.44 / 0.025) — same sign under
all three correctors, never below p < 0.01. Anchor −2.74, p = 1×10⁻³⁰.

## 20 Sept: seed repeats (3 seeds for the two endpoint configurations)

| Seed | CRNN-B | CRNN-LX | Δ | McNemar p |
|---|---:|---:|---:|---:|
| 42 (local RTX 4070) | 81.64 | 81.95 | +0.32 | 0.132 |
| 123 (Kaggle T4) | 81.93 | 81.43 | −0.49 | **0.024** |
| 456 (Kaggle T4) | 81.66 | 82.44 | +0.78 | **3.9×10⁻⁴** |
| **mean ± SD** | **81.74 ± 0.16** | **81.94 ± 0.50** | **+0.20 ± 0.64** | 0.64 (paired t) |

Two of three per-seed tests are "significant" in **opposite** directions, and
CRNN-LX alone spans 1.00 pp across seeds — larger than any augmentation effect
measured. The conventional-pipeline effect (−2.74 pp, p = 1×10⁻³⁰) is an order
of magnitude above that noise floor. Weights: `brht25/seed1-output`,
`brht25/seed2-output` (public); labels verified byte-identical to ours.

## 15 Sept (evening): real trigram with line context

| Post-correction on the CRNN-LX optical model | test WA | CER |
|---|---:|---:|
| none (greedy) | 78.83 | 8.60 |
| 239K lexicon + add-one unigram prior (old corrector) | 80.74 | 9.09 |
| + Kneser-Ney unigram, IAM | 80.74 | 9.09 |
| + KN trigram with line context, IAM | 80.84 | 9.07 |
| + KN unigram, IAM + Brown | 81.32 | 8.74 |
| **+ KN trigram with line context, IAM + Brown (CRNN-LX)** | **81.66** | **8.58** |

- Gain over the unigram prior: +0.85 to +0.96 pp on **all six** optical models
  (per-model `KN3-left` entries in `results/ablation_viterbi.json`); α = 7
  selected on validation for each.
- Oracle (reference transcriptions as context, diagnostic only): 81.73.
- Leakage: 0.63% of test 5-grams, 0.11% of 6-grams, 0 of 8-grams occur in
  Brown; longest shared run 7 words, all idioms (`results/brown_leakage.json`).
- Model verified numerically: conditional distributions sum to 1 over the
  271,303-type vocabulary (`cloud/kn_trigram_selftest.py`).
- **Table 2 with this corrector:** CRNN-LX − CRNN-B = +0.44 pp, **p = 0.025**
  (not significant at the paper's p < 0.01, nor after Bonferroni over five
  comparisons); anchor −2.41 pp, p = 8×10⁻²⁷. With the unigram prior the same
  pair was +0.38 pp, p = 0.060. The paper says so explicitly; seed repeats
  decide it.

The sections below describe the state before this change (unigram prior).

Every number in the paper comes from **one deterministic evaluation pass**
(`cloud/ablation_lexicon_all.py`, fp32): five optical models x five
post-corrections against identical inputs. Two independent runs agree on all
20,310 hypotheses. Earlier numbers mixed two evaluation paths (the training
script's own test loop and the lexicon-ablation script) that differed by up to
9 words, and used mixed-precision inference, which is not bit-reproducible.

**Two findings, both measured under identical conditions (same data, network,
schedule, machine, code revision):**

1. **The proposed augmentation contributes nothing measurable — and the
   experiment proves it can see augmentation.** Five configurations that keep
   the conventional pipeline (affine, narrow photometric, noise, gamma, random
   erasing) and differ only in the three proposed transforms reach
   80.33–80.74% WA; no pairwise McNemar test is significant (p ≥ 0.060).
   Switching the conventional pipeline off (`--aug-mode none`, sixth
   configuration) costs **−2.49 pp** (80.36 → 77.87) at **p = 2×10⁻²⁷**, and
   −4.43 pp lexicon-free (78.76 → 74.33). So the null result is **saturation**,
   not an insensitive experiment — the anchor is a positive control.
   The earlier "+6.48 pp" claim was an artefact of a truncated label file
   (39% of IAM) and mismatched pipelines.
2. **Post-correction has two ingredients and they do different jobs.**
   *Coverage* decides the sign: the 7,173-word training vocabulary *hurts*
   (−2.4 to −2.7 pp on every model), the 239,126-type extended list *helps*
   (+0.3 to +0.7 pp). The *frequency prior* that ranks candidates within edit
   distance then supplies most of the magnitude (+1.1 to +1.3 pp) and is worth
   twice as much on the large lexicon as on the small one, because a larger
   candidate set makes the ranking matter more. Together: +1.5 to +1.9 pp,
   while CER rises 8.60 → 9.09.

Plus: the common elastic-deformation parameterisation (normalised blur × α∈[2,5])
displaces pixels by only 0.02–0.04 px RMS per axis at the 64×256 working
resolution (max 0.19 px) — a no-op; the paper uses a unit-RMS field × 1–3 px.

## Model naming (fixed — use these names everywhere)

The WA column below is the 15 Sept state (unigram corrector). Current numbers
are in the tables at the top of this file and in the paper.

| Name | wide photometric | elastic | morph | test WA |
|---|:---:|:---:|:---:|---:|
| `no augmentation` (anchor) | — conventional pipeline off too — | | | 77.87 |
| `CRNN-B` (baseline) | no | no | no | 80.36 |
| `+ wide photometric` | yes | no | no | 80.33 |
| `+ elastic` | yes | yes | no | 80.67 |
| `+ morphological` | yes | no | yes | 80.64 |
| **`CRNN-LX`** (proposed) | yes | yes | yes | **80.74** |
| `CRNN-G` | = CRNN-LX optical model, greedy CTC, no lexicon | | | 78.83 |

CRNN-S / CRNN-M no longer appear (their numbers came from the truncated data).

## Files

- `paper.tex` — LaTeX source, **IJPRAI journal format** (`ws-ijprai`, single
  column, 18 pages)
- `paper_ieee.tex` — the earlier IEEEtran conference version (two column,
  8 pages), kept for reference. It compiles on its own. Its title is kept in
  sync, but its body predates the 15 Sept revision below.
- `ws-ijprai.cls`, `ws-ijprai.bst` — World Scientific class and bibliography
  style, from the publisher's `ijprai-2e` package
- `references.bib` — 21 entries (added: Simard 2003, Wigington 2017, Mondal 2022)
- `generate_figures.py` — regenerates every figure from `Model_abl_*/`
- `figures/` — 4 vector PDFs

## Figures

| File | Fig. | Content | Data source |
|---|---|---|---|
| `fig0_pipeline.pdf` | 1 | End-to-end system diagram; the two ablated stages are starred | drawn |
| `fig1_augmentation_grid.pdf` | 2 | 8 transforms on one IAM crop ("meeting"), elastic at 2 px RMS | real IAM crop + cv2 |
| `fig2_training_curves.pdf` | 3 | Validation WA + training loss, **6** configurations | `Model_abl_*/training_history.json` |
| `fig4_lexicon_decomposition.pdf` | 4 | **Headline finding**: ΔWA vs lexicon size, with and without the unigram frequency prior | `results/ablation_lexicon5_all.json` |
| `fig3_confusion_topk.pdf` | 5 | Top-10 character substitutions of CRNN-LX | `results/preds_det/preds_full.csv` |

Three defects were fixed in this pass: the pipeline box still carried the
**truncated** split (31.3K/1.6K/5.3K instead of 48.0K/7.2K/20.3K) and marked
augmentation as a contribution; the augmentation grid used whatever PNG sorted
first on disk, an illegible single-letter fragment; and `colors` had five
entries after `ABL_MODES` grew to six, so `zip()` silently dropped the proposed
system from its own training-curve figure.

```bash
python makale/generate_figures.py
```

## Build

```bash
cd makale && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
Overleaf: upload `makale_overleaf.zip` (repo root), compiler pdfLaTeX. The zip
carries `ws-ijprai.cls` and `ws-ijprai.bst`, so no template needs installing.

### What the IJPRAI conversion changed

Content is unchanged; only what the journal mandates was touched.

| | IEEEtran (`paper_ieee.tex`) | IJPRAI (`paper.tex`) |
|---|---|---|
| Layout | two column, 8 pages | single column, 18 pages |
| Authors | `\IEEEauthorblockN/A` | one `\author`+`\address` pair each |
| Front matter | — | `\markboth`, `\catchline`, `\history` |
| Abstract | 306 words, contained math | **199 words**, no math, no citations |
| Keywords | `IEEEkeywords` | `\keywords{a; b; c.}` |
| Lists | `itemize` | `itemlist` |
| Tables | `\caption` + `\resizebox` | `\tbl{caption\label{}}{body}`, `\colrule`/`\botrule`, `tabfootnote` |
| Figures | `\caption` | `+ \alttext{}`, widths in cm |
| Citations | `[1]` bracketed, citation order | superscript (`overcite`), **alphabetical** |
| Spelling | British | **American** (44 words changed) |

Two latent `references.bib` defects surfaced under the stricter style and were
fixed: `RWTH Aachen University` printed as the person "R. A. University"
(needed double braces) and `Arthur Flôr de Sousa Neto` lost its surname
particle (needed `{de Sousa Neto}` as the family name).

## Paper structure (8 pages)

1. **Introduction** — HTR/CTC/word-level concepts, 4 contributions
2. **Related Work** — 4 groups, "what we adopt, where we differ"
3. **Proposed System** — pipeline, data (47,997 / 7,205 / 20,310), naming, model, augmentation (incl. elastic no-op finding), lexicon post-correction (coverage 84.8% / 94.0%)
4. **Experimental Setup** — WA/CER/Wilson/McNemar (eq. 1–4), reproducibility (local RTX 4070, single seed)
5. **Results** — augmentation ablation (Table II), lexicon ablation (Table III), prior work with a lexicon column (Table IV), error analysis
6. **Discussion** — why augmentation doesn't help, coverage vs. the frequency prior, same-protocol comparison with Sueiras et al. 2018 (+2.6 pp lexicon-free, +4.5 pp with lexicon), HWRCNet note, threats to validity
7. **Conclusion**

## Every number is verifiable

| Claim | Source |
|---|---|
| Table II (5 configs WA/CER/CI) | `results/ablation_lexicon5_all.json`, row "extended lexicon + n-gram" |
| Table II McNemar p | `results/ablation_lexicon5_all.json`, key `mcnemar_vs_narrow` (exact binomial on `results/preds_det/preds_<mode>.csv`) |
| best val WA (epoch) | `Model_abl_<mode>/training_history.json` |
| anchor row (−2.49 pp, p = 2×10⁻²⁷) | `Model_abl_none/`, scored on the same deterministic path |
| Table III (post-correction ablation) | `results/ablation_lexicon5_all.json`, model `full` |
| coverage 84.8% / 94.0% | computed from `aachen_splits/{train,test}_words.txt` + NLTK |
| error analysis counts, Fig. 4 | `results/preds_det/preds_full.csv` (Levenshtein alignment) |
| external systems | read from the cited papers; Sueiras 2018 (WER 23.8 / CER 8.8, lexicon-free) cross-checked in Dutta 2018 Tab. III, Kang 2021 Tab. 7, Kass & Vats 2022 Tab. 5, Mondal 2022 Tab. 1; the "Dutta 77.14" row is HWRCNet's own re-training (Rajesh 2022 Tab. 2), Dutta's own figure is 12.61 % WER |

Full report with all tables: `../results/ABLATION_SONUC.md`.


## Advisor's review of 15 September 2026 — all seven points closed

The wording below is our itemisation of the points, not a verbatim quote of
the messages. "Then" is what the 15 Sept revision did; "now" is where the
paper stands after the trigram and seed work of 16-20 Sept.

| # | Advisor's point | Then | Now |
|---|---|---|---|
| 1 | §6.1 heading asserted a conclusion | → *Interpretation of the Augmentation Results* | unchanged |
| 2 | §6.3 heading named one paper | → *Comparison with Previous Word-Level HTR Systems* | unchanged |
| 3 | Title too assertive | → *Decomposing Lexicon-Assisted Correction for Isolated Handwritten Word Recognition on IAM: Effects of Lexicon Coverage and Frequency-Based Ranking* | changed again, to *… for Word-Level Handwritten Text Recognition on IAM: Lexicon Coverage, Frequency Ranking and Line Context* |
| 4 | "trigram" without word context | Renamed to **unigram frequency prior** throughout, because `score_word` was always called with `prev_words=None`; §3.5 said so and gave the sparsity figure (34 % of test tokens had a seen trigram context) | **fixed at the root**: a real interpolated Kneser-Ney trigram with line context now ranks the candidates (IAM + Brown, whole-line Viterbi). The unigram prior is kept as the no-context reference row. The 34 % figure was dropped with it. |
| 5 | Single seed | Wording softened in the abstract, contribution 2, §5.1, §6.1, §6.3 and the Conclusion; seed repeats planned | **done**: CRNN-B and CRNN-LX × seeds 42/123/456 → **Table 3**. The difference changes sign (+0.32 / −0.49 / +0.78 pp; mean +0.20 ± 0.64, paired p = 0.64) and one configuration spans 1.00 pp, so the augmentation comparison is reported as unresolved. |
| 6 | Table 2 ΔWA ≠ numbers in text | Δ_B (vs CRNN-B) **and** Δ_P (vs +wide photometric), caption states both come from unrounded accuracies | unchanged; every cell is now machine-checked by `cloud/verify_paper_numbers.py` |
| 7 | "2–4 pp" | → "1.8–3.9 pp below the later attention-based systems" | now 0.6–2.7 pp with line context and 1.8–3.9 pp without it; both are in the text |

Three further corrections found while checking Section 3 against the code:
morphological perturbation fires with probability **0.15**, not 0.3 (half of
the 0.3 draws pick a 1×1 no-op kernel); early stopping resets on an
improvement of validation loss **or** WA; the legacy elastic amplitude is
0.02–0.04 px RMS per axis (max 0.19 px), measured with the released code at
64×256.

## Author TODO before submission

- [x] `\author{}` block filled: Nur Banu Oğur (corresponding, nbogur@sakarya.edu.tr), Rıdvan Dursun, Berhat Yeşilyurt; Dept. of Software Engineering, Faculty of Computer and Information Sciences, Sakarya University
- [ ] Confirm Berhat's institutional e-mail (derived from the `ad.soyad@ogr.sakarya.edu.tr` pattern, not verified)
- [ ] Re-verify every bib entry against the publisher page
- [ ] One English proofreading pass
- [ ] Decide venue; switch `\documentclass` if needed
- [x] **Seed repeats** (advisor, 5a): CRNN-B and CRNN-LX with seeds 123 and 456, trained on Kaggle (T4) and scored here with the final corrector — **Table 3** of the paper. The difference changes sign across seeds (+0.32 / −0.49 / +0.78 pp; mean +0.20 ± 0.64, paired p = 0.64) and one configuration spans 1.00 pp, so the augmentation comparison is reported as unresolved. Source: `results/ablation_viterbi_seeds.json`
- [x] Elastic amplitude in the seed runs confirmed from Berhat's Kaggle logs: both print `Elastic : alpha 1-3  RMS px (fixed)`, i.e. the corrected amplitude, not the legacy no-op. §4 records the check.
- [ ] Optional: WBS / TTA / ensembling re-evaluated on the full data (removed from the paper; old numbers were from truncated data)

## Repo

https://github.com/Ridvan013/CRNN-Handwriting-Recognition
(branch `feature/aachen-v3-extended-trigram`)
