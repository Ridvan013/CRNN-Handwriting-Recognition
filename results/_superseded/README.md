# Superseded results (do not cite)

These files come from `cloud/ablation_lexicon.py` before three fixes:

1. inference ran under AMP (fp16), which is not bit-reproducible -- two runs
   of the same checkpoint disagreed on 2 of 20,310 words;
2. the "edit distance only" configuration broke ties by Python `set`
   iteration order, which moved WA by 0.41 pp between runs;
3. the extended lexicon *without* n-gram rescoring was never measured, so
   the contribution of the n-gram prior at the operative lexicon size was
   unknown (it is +1.1 to +1.3 pp, not the ~+0.2 pp the 7K lexicon suggests).

The current numbers -- six optical models x five post-corrections, fp32,
deterministic, one evaluation pass -- are in
`results/ablation_lexicon5_all.json`, with per-word predictions in
`results/preds_det/`. The paper uses only those.

- `ablation_trigram_all_nondet.json`, `ablation_viterbi_grid3-10.json` (16 Eylül):
  cuDNN deterministik bayrakları eklenmeden önceki koşular (süreçler arası
  1 kelime oynuyordu) ve Viterbi'nin dar {3,5,7,10} α ızgarasıyla ilk denemesi.
  Geçerli kaynak: `results/ablation_viterbi.json` (Tablo 2, Tablo 3 satır 10-13)
  ve `results/ablation_trigram.json` (Tablo 3 satır 5-9).
