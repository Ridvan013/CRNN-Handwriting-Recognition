"""Numerical self-test of cloud/kn_trigram.py: normalisation, back-off, context reset, determinism.

Usage:  python cloud/kn_trigram_selftest.py      (CPU, about two minutes)
"""
import sys, math, random
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cloud"))
from kn_trigram import KNTrigram, iam_line_segments, brown_sentences, ContextCorrector
from trigram_lm import TrigramLanguageModel

train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
lex = TrigramLanguageModel(train_words, use_nltk_extension=True)
segs = iam_line_segments(train_words)
kn = KNTrigram(segs + brown_sentences(), discount=0.75, extra_vocab=lex.vocabulary)
print(kn.describe())

# vocabulary = LM types + lexicon
vocab = set(kn.cont1) | set(lex.vocabulary)
# words that only ever start a segment have cont1 = 0 but are still in V
V_all = set()
for s in segs + brown_sentences():
    V_all.update(s)
vocab |= V_all
assert len(vocab) == kn.V, (len(vocab), kn.V)
vocab = sorted(vocab)

def check(name, f):
    s = sum(f(w) for w in vocab)
    ok = abs(s - 1.0) < 1e-9
    print(f"  {name:<40s} sum = {s:.12f}  {'OK' if ok else 'FAIL'}")
    assert ok

print("normalisation over V =", len(vocab))
check("P(w)", lambda w: kn.p_uni(w))
for w2 in ["the", "of", "in", ",", "Mr", "zzzunknownzzz"]:
    check(f"P(w | {w2!r})", lambda w, w2=w2: kn.p_bi(w, w2))
for w1, w2 in [("in", "the"), ("one", "of"), (",", "and"), ("zz", "the"), ("the", "zz")]:
    check(f"P(w | {w1!r} {w2!r})", lambda w, w1=w1, w2=w2: kn.p_tri(w, w1, w2))

# back-off: unseen context must equal the lower order exactly
assert kn.p_tri("house", "zz", "the") == kn.p_bi("house", "the")
assert kn.p_tri("house", None, "the") == kn.p_bi("house", "the")
assert kn.p_tri("house", "in", None) == kn.p_uni("house")
assert kn.p_bi("house", "zzzunknownzzz") == kn.p_uni("house")
print("back-off to lower order: OK")

# context should matter: P(States | United) >> P(States)
print(f"  P(States)={kn.p_uni('States'):.2e}  P(States|United)={kn.p_bi('States','United'):.2e}  "
      f"P(States|the,United)={kn.p_tri('States','the','United'):.2e}")
assert kn.p_bi("States", "United") > 50 * kn.p_uni("States")

# every candidate has positive probability (floor)
assert kn.p_uni("Qzxq") > 0 and math.isfinite(kn.logp("Qzxq", "in", "the"))
print("positive floor for unseen words: OK")

# corrector: context reset at line start / gap, order independence, determinism
cc = ContextCorrector(lex, kn)
rows = [("L1", 0, "Tne"), ("L1", 1, "United"), ("L1", 2, "Stares"), ("L1", 4, "hous"), ("L2", 0, "Stares")]
o1 = cc.correct_lines(rows, 5.0, True)
perm = list(range(len(rows))); random.Random(0).shuffle(perm)
o2 = cc.correct_lines([rows[i] for i in perm], 5.0, True)
assert [o2[perm.index(i)] for i in range(len(rows))] == o1, "order dependence!"
o3 = cc.correct_lines(rows, 5.0, True)
assert o1 == o3
print("corrector outputs:", list(zip([r[2] for r in rows], o1)))
# word after a gap (idx 4) and the first word of L2 must be scored WITHOUT context
assert o1[3] == cc.correct_one("hous", None, None, 5.0, True)
assert o1[4] == cc.correct_one("Stares", None, None, 5.0, True)
# the in-context "Stares" after "United" should become "States"
assert o1[2] == "States", o1[2]
print("context reset at line start / gap, order-independent, deterministic: OK")
print("ALL SELF-TESTS PASSED")
