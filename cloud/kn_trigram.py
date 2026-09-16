#!/usr/bin/env python3
"""
Interpolated Kneser-Ney trigram language model + line-context post-corrector.

Why this exists
---------------
The paper's post-corrector ranks lexicon candidates with a unigram frequency
prior: ``TrigramLanguageModel.correct_word`` is always called without previous
words, so its bigram/trigram branches never run.  This module is the *real*
thing: a properly smoothed trigram (Chen & Goodman's interpolated Kneser-Ney)
that scores each candidate given the words that precede it on the same text
line.  The context is the recognizer's OWN OUTPUT for the preceding words,
never the ground truth -- using the neighbours' transcriptions would leak the
test labels.

Tokens are case-sensitive (word accuracy is case-sensitive) and no sentence
boundary symbols are used: IAM lines are physical lines, not sentences, so a
line start carries no linguistic boundary.  Context is reset to "unknown"
(unigram scoring) at a line start and wherever the preceding word index is
missing from the split (words whose segmentation IAM flags as not ``ok`` are
not part of the Aachen word lists).

Interpolated KN, orders 3/2/1 (Chen & Goodman 1999, eq. for KN-interp):

    P(w3|w1 w2) = max(c(w1 w2 w3) - D, 0) / c(w1 w2 .)
                  + D * N1+(w1 w2 .) / c(w1 w2 .) * P(w3|w2)
    P(w3|w2)    = max(N1+(. w2 w3) - D, 0) / N1+(. w2 .)
                  + D * N1+(w2 .) / N1+(. w2 .) * P(w3)
    P(w3)       = (N1+(. w3) + beta) / (N1+(. .) + beta * V)

with D = 0.75 and a small additive floor (beta = 1 over the union of the LM
vocabulary and the correction lexicon) so that lexicon words never seen in the
corpus keep a positive probability.  If a context was never observed the model
falls to the next lower order.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class KNTrigram:
    def __init__(self, sentences: Iterable[Sequence[str]], discount: float = 0.75,
                 extra_vocab: Optional[Iterable[str]] = None, beta: float = 1.0):
        self.D = float(discount)
        self.beta = float(beta)
        c3: Counter = Counter()
        c2: Counter = Counter()
        c1: Counter = Counter()
        for s in sentences:
            toks = list(s)
            for t in toks:
                c1[t] += 1
            for i in range(1, len(toks)):
                c2[(toks[i - 1], toks[i])] += 1
            for i in range(2, len(toks)):
                c3[(toks[i - 2], toks[i - 1], toks[i])] += 1
        self.c3 = c3
        # trigram-context totals and "types following (w1,w2)"
        self.ctx2: Counter = Counter()          # c(w1 w2 .)
        self.follow2: Counter = Counter()       # N1+(w1 w2 .)
        cont2: Counter = Counter()              # N1+(. w2 w3)
        for (w1, w2, w3), c in c3.items():
            self.ctx2[(w1, w2)] += c
            self.follow2[(w1, w2)] += 1
            cont2[(w2, w3)] += 1
        self.cont2 = cont2
        self.contmid: Counter = Counter()       # N1+(. w2 .)
        self.follow1: Counter = Counter()       # N1+(w2 .)
        for (w2, w3), c in cont2.items():
            self.contmid[w2] += c
            self.follow1[w2] += 1
        self.cont1: Counter = Counter()         # N1+(. w3)
        for (w2, w3) in c2:
            self.cont1[w3] += 1
        self.n_bigram_types = len(c2)           # N1+(. .)
        vocab = set(c1)
        if extra_vocab is not None:
            vocab |= set(extra_vocab)
        self.V = len(vocab)
        self.n_tokens = sum(c1.values())
        self.n_types = len(c1)
        self._uni_cache: Dict[str, float] = {}

    # ---- probabilities --------------------------------------------------
    def p_uni(self, w: str) -> float:
        p = self._uni_cache.get(w)
        if p is None:
            p = (self.cont1.get(w, 0) + self.beta) / (self.n_bigram_types + self.beta * self.V)
            self._uni_cache[w] = p
        return p

    def p_bi(self, w3: str, w2: str) -> float:
        tot = self.contmid.get(w2, 0)
        if tot == 0:
            return self.p_uni(w3)
        num = max(self.cont2.get((w2, w3), 0) - self.D, 0.0)
        lam = self.D * self.follow1.get(w2, 0) / tot
        return num / tot + lam * self.p_uni(w3)

    def p_tri(self, w3: str, w1: Optional[str], w2: Optional[str]) -> float:
        if w2 is None:
            return self.p_uni(w3)
        if w1 is None:
            return self.p_bi(w3, w2)
        tot = self.ctx2.get((w1, w2), 0)
        if tot == 0:
            return self.p_bi(w3, w2)
        num = max(self.c3.get((w1, w2, w3), 0) - self.D, 0.0)
        lam = self.D * self.follow2.get((w1, w2), 0) / tot
        return num / tot + lam * self.p_bi(w3, w2)

    def logp(self, w3: str, w1: Optional[str] = None, w2: Optional[str] = None) -> float:
        return math.log(self.p_tri(w3, w1, w2))

    def describe(self) -> str:
        return (f"KN trigram: {self.n_tokens:,} tokens, {self.n_types:,} types, "
                f"{len(self.c3):,} trigrams, {self.n_bigram_types:,} bigrams, "
                f"V={self.V:,}, D={self.D}")


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def iam_line_segments(words_file: str) -> List[List[str]]:
    """Contiguous runs of consecutive words on the same IAM line, in order.

    A run is broken wherever a word index is missing (the Aachen lists keep
    only words whose segmentation is flagged ``ok``), so no n-gram is ever
    counted across a gap.  Transcriptions with internal spaces (e.g. "B B C")
    are joined as the recognizer sees them.
    """
    per_line: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with open(words_file, encoding="utf-8") as fh:
        for l in fh:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            parts = l.split()
            if len(parts) < 9:
                continue
            wid = parts[0]
            line_id = "-".join(wid.split("-")[:3])
            idx = int(wid.split("-")[3])
            word = "".join(parts[8:])
            per_line[line_id].append((idx, word))
    segments: List[List[str]] = []
    for line_id, items in per_line.items():
        items.sort()
        run: List[str] = []
        prev = None
        for idx, w in items:
            if prev is not None and idx != prev + 1:
                if run:
                    segments.append(run)
                run = []
            run.append(w)
            prev = idx
        if run:
            segments.append(run)
    return segments


def brown_sentences() -> List[List[str]]:
    """Brown corpus sentences (57,340 sentences, 1,161,192 tokens), downloaded
    through NLTK on first use (needs Internet, e.g. Kaggle 'Internet ON')."""
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    sents = [list(s) for s in brown.sents()]
    assert len(sents) == 57340, f"unexpected Brown corpus size: {len(sents)} sentences"
    return sents


# ---------------------------------------------------------------------------
# Line-context corrector
# ---------------------------------------------------------------------------
class ContextCorrector:
    """Same acceptance rule and candidate set as TrigramLanguageModel.correct_word
    (in-lexicon hypotheses are accepted unchanged; otherwise lexicon entries
    within Levenshtein distance 1 for |w|<=4, else 2), but the candidates are
    ranked by  log P_KN(candidate | previous two output words) - alpha * dist.
    Ties are broken exactly as in correct_word: shorter candidate length first,
    then lexicographic order within a length bucket."""

    def __init__(self, lexicon_lm, kn: KNTrigram):
        self.lex = lexicon_lm          # TrigramLanguageModel (for vocabulary + buckets)
        self.kn = kn
        self._cands: Dict[str, List[Tuple[str, int]]] = {}
        self._ensure_buckets()

    def _ensure_buckets(self):
        lex = self.lex
        by_len = lex.__dict__.get("_vocab_by_len")
        if by_len is None or lex.__dict__.get("_vocab_by_len_n") != len(lex.vocabulary):
            by_len = {}
            for v in lex.vocabulary:
                by_len.setdefault(len(v), []).append(v)
            for k in by_len:
                by_len[k].sort()
            lex._vocab_by_len = by_len
            lex._vocab_by_len_n = len(lex.vocabulary)
        self.by_len = by_len

    def in_lexicon(self, w: str) -> bool:
        return w in self.lex.vocabulary or w.lower() in self.lex.vocabulary_lower

    def candidates(self, w: str, max_dist: Optional[int] = None) -> List[Tuple[str, int]]:
        """(candidate, distance) within the edit bound, in tie-break order
        (shorter length first, lexicographic within a length).  The default
        bound is the paper's: 1 for |w| <= 4, else 2.  Independent of context
        and alpha, so memoised per (hypothesis, bound)."""
        if max_dist is None:
            max_dist = 1 if len(w) <= 4 else 2
        key = (w, max_dist)
        c = self._cands.get(key)
        if c is not None:
            return c
        import numpy as np
        out: List[Tuple[str, int]] = []
        for L in range(len(w) - max_dist, len(w) + max_dist + 1):
            bucket = self.by_len.get(L, ())
            if not bucket:
                continue
            dists = self.lex._bucket_distances(w, L, bucket, max_dist)
            for i in np.nonzero(dists <= max_dist)[0]:
                out.append((bucket[int(i)], int(dists[int(i)])))
        self._cands[key] = out
        return out

    def correct_one(self, w: str, h1: Optional[str], h2: Optional[str], alpha: float,
                    use_context: bool) -> str:
        if self.in_lexicon(w):
            return w
        best, best_score = w, None
        for cand, dist in self.candidates(w):
            lp = self.kn.logp(cand, h1, h2) if use_context else self.kn.logp(cand)
            score = lp - alpha * dist
            if best_score is None or score > best_score:     # strict: keeps first on ties
                best, best_score = cand, score
        return best

    def correct_lines(self, rows: Sequence[Tuple[str, int, str]], alpha: float,
                      use_context: bool,
                      context_source: Optional[Sequence[str]] = None) -> List[str]:
        """rows: (line_id, word_idx, hypothesis) in any order; returns outputs
        aligned with rows.  Words of a line are processed in reading order and
        each output becomes context for the next word.  If `context_source`
        is given (e.g. ground truth, for an ORACLE diagnostic), it replaces the
        model's own outputs as context; never use that for a reported result."""
        order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
        out: List[Optional[str]] = [None] * len(rows)
        prev_line, prev_idx = None, None
        h1: Optional[str] = None      # word two back
        h2: Optional[str] = None      # word one back
        for i in order:
            line_id, idx, hyp = rows[i]
            if line_id != prev_line or prev_idx is None or idx != prev_idx + 1:
                h1, h2 = None, None            # line start or gap: no context
            o = self.correct_one(hyp, h1, h2, alpha, use_context)
            out[i] = o
            ctx_word = context_source[i] if context_source is not None else o
            h1, h2 = h2, ctx_word
            prev_line, prev_idx = line_id, idx
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Two-sided (whole-line) correction: exact second-order Viterbi
# ---------------------------------------------------------------------------
def line_segments(rows: Sequence[Tuple[str, int, str]]) -> List[List[int]]:
    """Row indices grouped into contiguous runs (same line, consecutive word
    indices), in reading order.  Runs are independent: no n-gram crosses a
    line start or a missing word."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    segs: List[List[int]] = []
    prev_line, prev_idx = None, None
    for i in order:
        line_id, idx, _ = rows[i]
        if segs and line_id == prev_line and idx == prev_idx + 1:
            segs[-1].append(i)
        else:
            segs.append([i])
        prev_line, prev_idx = line_id, idx
    return segs


class LineViterbiCorrector(ContextCorrector):
    """Chooses the word sequence of a whole line jointly.

    Every word crop contributes a column of candidates (word, edit distance
    from the recognizer's hypothesis).  The decoder returns the sequence
    w_1..w_n that maximises

        sum_i  log P_KN(w_i | w_{i-2}, w_{i-1})  -  alpha * d_i

    over all paths through the columns.  Because w_i enters the terms of
    w_{i+1} and w_{i+2} as well as its own, each choice is conditioned on the
    words to its LEFT and to its RIGHT -- the joint probability of the line,
    not a left-to-right greedy decision.  The maximisation is exact (second-
    order Viterbi, state = last two columns), deterministic (strict '>' keeps
    the first maximiser; columns are in a fixed order) and independent of the
    order in which lines are supplied.

    Column construction (options are selected on the validation set):
      * hypothesis outside the lexicon: its lexicon candidates within the
        paper's edit bound; if there are none, the hypothesis itself.
        keep_oov=True also keeps the hypothesis as a candidate (d = 0), so a
        correct out-of-lexicon word such as a proper noun can survive when
        the line supports it; its probability is the model's unseen-word floor.
      * hypothesis inside the lexicon: kept unchanged, unless real_word=True,
        in which case lexicon entries within distance `rw_dist` compete with
        it (d = 0 for the hypothesis), so real-word errors ("form"/"from")
        become correctable.
      * columns are pruned to the `topk` candidates with the best
        context-free score  log P(w) - alpha * d  (the hypothesis itself,
        when kept, is always retained).
    """

    def __init__(self, lexicon_lm, kn: KNTrigram, topk: int = 10,
                 memo_limit: int = 5_000_000):
        super().__init__(lexicon_lm, kn)
        self.topk = int(topk)
        self.memo_limit = memo_limit
        self._lp: Dict[Tuple[Optional[str], Optional[str], str], float] = {}

    def _logp(self, w3: str, w1: Optional[str], w2: Optional[str]) -> float:
        key = (w1, w2, w3)
        v = self._lp.get(key)
        if v is None:
            v = self.kn.logp(w3, w1, w2)
            if len(self._lp) < self.memo_limit:
                self._lp[key] = v
        return v

    def column(self, w: str, alpha: float, real_word: bool = False,
               keep_oov: bool = False, rw_dist: int = 1) -> List[Tuple[str, int]]:
        if self.in_lexicon(w):
            if not real_word:
                return [(w, 0)]
            cands = [(c, d) for c, d in self.candidates(w, rw_dist) if c != w]
            keep_self = True
        else:
            cands = self.candidates(w)
            if not cands:
                return [(w, 0)]
            keep_self = keep_oov
        # stable sort: ties keep the candidate generator's order
        ranked = sorted(cands, key=lambda cd: -(self._logp(cd[0], None, None) - alpha * cd[1]))
        k = max(self.topk - (1 if keep_self else 0), 0)
        return ([(w, 0)] if keep_self else []) + ranked[:k]

    def sequence_score(self, words: Sequence[str], dists: Sequence[int], alpha: float) -> float:
        s = 0.0
        for i, (w, d) in enumerate(zip(words, dists)):
            w1 = words[i - 2] if i >= 2 else None
            w2 = words[i - 1] if i >= 1 else None
            s += self._logp(w, w1, w2) - alpha * d
        return s

    def viterbi(self, cols: Sequence[Sequence[Tuple[str, int]]], alpha: float) -> Tuple[List[int], float]:
        """Exact argmax over paths; returns (chosen index per column, score)."""
        n = len(cols)
        # state key (j, k): j = index in column i-1 (-1 before the start), k = index in column i
        first: Dict[Tuple[int, int], Tuple[float, Optional[Tuple[int, int]]]] = {}
        for k, (c, d) in enumerate(cols[0]):
            first[(-1, k)] = (self._logp(c, None, None) - alpha * d, None)
        hist = [first]
        for i in range(1, n):
            prev = hist[-1]
            cur: Dict[Tuple[int, int], Tuple[float, Optional[Tuple[int, int]]]] = {}
            c_prev = cols[i - 1]
            c_pp = cols[i - 2] if i >= 2 else None
            for k, (c, d) in enumerate(cols[i]):
                pen = alpha * d
                for (h, j), (s, _) in prev.items():
                    w1 = c_pp[h][0] if h >= 0 else None
                    sc = s + self._logp(c, w1, c_prev[j][0]) - pen
                    old = cur.get((j, k))
                    if old is None or sc > old[0]:
                        cur[(j, k)] = (sc, (h, j))
            hist.append(cur)
        best_key, best = None, None
        for key, (s, _) in hist[-1].items():
            if best is None or s > best:
                best_key, best = key, s
        idx = [0] * n
        key = best_key
        for i in range(n - 1, -1, -1):
            idx[i] = key[1]
            key = hist[i][key][1]
        return idx, best  # type: ignore[return-value]

    def decode_lines(self, rows: Sequence[Tuple[str, int, str]], alpha: float,
                     real_word: bool = False, keep_oov: bool = False,
                     rw_dist: int = 1) -> List[str]:
        """rows: (line_id, word_idx, hypothesis) in any order; outputs aligned with rows."""
        out: List[Optional[str]] = [None] * len(rows)
        for seg in line_segments(rows):
            cols = [self.column(rows[i][2], alpha, real_word, keep_oov, rw_dist) for i in seg]
            if all(len(c) == 1 for c in cols):
                for i, c in zip(seg, cols):
                    out[i] = c[0][0]
                continue
            idx, _ = self.viterbi(cols, alpha)
            for i, c, k in zip(seg, cols, idx):
                out[i] = c[k][0]
        return out  # type: ignore[return-value]
