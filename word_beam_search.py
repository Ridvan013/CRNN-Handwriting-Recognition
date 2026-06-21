"""
Word Beam Search Decoder (Scheidl, Fiel, Sablatnig, ICFHR 2018)
Lexicon- and n-gram-aware CTC decoder for word-level handwriting recognition.

Implementation tailored to single-word images (no whitespace handling).
Supports two modes:
    - "Words": lexicon constraint only (prefix-tree pruning)
    - "NGrams": lexicon + character bigram language model during decoding

This decoder REPLACES post-hoc trigram correction by integrating the
language model INTO the CTC decoding step.

Reference: H. Scheidl, S. Fiel, R. Sablatnig, "Word Beam Search: A Connectionist
Temporal Classification Decoding Algorithm", ICFHR 2018, pp. 253-258.
"""
import math
from collections import Counter, defaultdict
from typing import Optional


# ============================================================
# Prefix Tree (Trie) for vocabulary
# ============================================================
class TrieNode:
    __slots__ = ('children', 'is_word')

    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    """Prefix tree of valid vocabulary words."""
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    @classmethod
    def from_vocabulary(cls, vocab):
        t = cls()
        for w in vocab:
            if w:
                t.add_word(w)
        return t


# ============================================================
# Character bigram language model (built from training words)
# ============================================================
class CharBigramLM:
    """Character bigram model with start/end tokens.
    P(c | prev) = (count(prev, c) + alpha) / (count(prev) + alpha * V)
    """
    START = '<s>'
    END = '</s>'

    def __init__(self, words, alpha: float = 0.1):
        self.alpha = alpha
        bigram = defaultdict(Counter)
        unigram = Counter()
        char_set = set()
        for w in words:
            chars = [self.START] + list(w) + [self.END]
            for i in range(len(chars) - 1):
                bigram[chars[i]][chars[i + 1]] += 1
                unigram[chars[i]] += 1
                char_set.add(chars[i])
                char_set.add(chars[i + 1])
        self.bigram = bigram
        self.unigram = unigram
        self.V = max(len(char_set), 1)

    def log_prob(self, prev_char: str, char: str) -> float:
        """log P(char | prev_char) with add-alpha smoothing."""
        num = self.bigram.get(prev_char, {}).get(char, 0) + self.alpha
        den = self.unigram.get(prev_char, 0) + self.alpha * self.V
        return math.log(num / den) if den > 0 else math.log(1.0 / self.V)


# ============================================================
# Word Beam Search Decoder
# ============================================================
class WordBeamSearchDecoder:
    """
    Lexicon-constrained CTC beam search decoder for word-level recognition.

    Args:
        char_list: ordered string of characters used by the CRNN
                   (blank token is implicitly index len(char_list))
        vocabulary: iterable of valid vocabulary words
        beam_width: number of beams to maintain per timestep
        mode: "Words" (lexicon only) or "NGrams" (lexicon + char bigram LM)
        lm_weight: weight on the LM log-probability (when mode="NGrams")
        lm_words: iterable of words used to build the character bigram LM
                  (defaults to vocabulary if None)
    """
    BLANK_PLACEHOLDER = None  # marker for "no last char emitted"

    def __init__(self,
                 char_list: str,
                 vocabulary,
                 beam_width: int = 25,
                 mode: str = "NGrams",
                 lm_weight: float = 0.7,
                 lm_words=None):
        self.char_list = char_list
        self.num_chars = len(char_list)
        self.blank_index = self.num_chars  # CTC blank
        self.beam_width = beam_width
        self.mode = mode
        self.lm_weight = lm_weight

        vocab_list = list(vocabulary)
        self.vocab_size = len(vocab_list)
        self.trie = Trie.from_vocabulary(vocab_list)

        if mode == "NGrams":
            if lm_words is None:
                lm_words = vocab_list
            self.char_lm = CharBigramLM(lm_words)
        else:
            self.char_lm = None

    # ---------------------------------------------------------
    # Beam state representation:
    #   beam = (prefix_str, trie_node, last_emit_idx_or_None, score)
    # We use a dict keyed by (prefix_str, last_emit_idx) for merging.
    # ---------------------------------------------------------

    def _expand_step(self, beams, log_probs_t):
        """One CTC time-step expansion with trie constraint."""
        # Convert log_probs_t to list for fast access (CPU)
        if hasattr(log_probs_t, 'tolist'):
            lp = log_probs_t.tolist()
        else:
            lp = list(log_probs_t)

        # Sort character indices by probability (highest first) for early pruning.
        # Use only top-(2*beam_width) candidates for speed (matches Scheidl impl).
        topk = sorted(range(len(lp)), key=lambda i: lp[i], reverse=True)
        topk = topk[: max(self.beam_width * 2, 10)]

        new_beams = {}
        prev_char_bigram = CharBigramLM.START  # used at start

        for prefix, node, last_idx, score in beams:
            # The "previous character" for bigram LM:
            # use last char of prefix, else START token
            prev_for_lm = prefix[-1] if prefix else CharBigramLM.START

            for char_idx in topk:
                char_lp = lp[char_idx]
                if char_lp == float('-inf'):
                    continue

                if char_idx == self.blank_index:
                    # Emit blank: prefix unchanged, last_idx -> None
                    key = (prefix, None)
                    new_score = score + char_lp
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                    # Also store node (same)
                    continue

                ch = self.char_list[char_idx]

                if char_idx == last_idx:
                    # Repeat (CTC collapse): prefix unchanged
                    key = (prefix, char_idx)
                    new_score = score + char_lp
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                    continue

                # New character emission: must extend valid trie prefix
                if ch not in node.children:
                    continue  # invalid extension - prune

                # LM bonus (character bigram)
                lm_bonus = 0.0
                if self.char_lm is not None:
                    lm_bonus = self.lm_weight * self.char_lm.log_prob(prev_for_lm, ch)

                new_prefix = prefix + ch
                key = (new_prefix, char_idx)
                new_score = score + char_lp + lm_bonus
                if key not in new_beams or new_beams[key] < new_score:
                    new_beams[key] = new_score

        # Reconstruct beam list with trie nodes (re-walk trie for each new prefix)
        # Note: For speed, we cache traversal by prefix length.
        # In practice prefixes share common roots so memo helps; here we just rewalk.
        scored = []
        for (pfx, last_idx), sc in new_beams.items():
            node = self._walk_trie(pfx)
            if node is None:
                continue  # safety - shouldn't happen
            scored.append((pfx, node, last_idx, sc))

        # Prune to top-beam_width
        scored.sort(key=lambda b: b[3], reverse=True)
        return scored[: self.beam_width]

    def _walk_trie(self, prefix: str):
        node = self.trie.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def decode_single(self, log_probs_seq):
        """Decode one CRNN output sequence.

        log_probs_seq: 2D tensor or array, shape [T, num_classes]
        Returns: best word string (may be empty if no path found)
        """
        # Initial beam: empty prefix at trie root, score 0, no last emit
        beams = [("", self.trie.root, None, 0.0)]

        T = log_probs_seq.shape[0] if hasattr(log_probs_seq, 'shape') else len(log_probs_seq)
        for t in range(T):
            beams = self._expand_step(beams, log_probs_seq[t])
            if not beams:
                return ""

        # Pick best beam that ends at a complete vocabulary word (preferred)
        complete_beams = [(pfx, sc) for (pfx, node, _, sc) in beams if node.is_word]
        if complete_beams:
            complete_beams.sort(key=lambda b: b[1], reverse=True)
            return complete_beams[0][0]

        # Fallback: best beam overall (even if mid-word)
        beams.sort(key=lambda b: b[3], reverse=True)
        return beams[0][0]

    def decode_batch(self, log_probs, input_lengths):
        """Decode a full batch.

        log_probs:    [T, B, num_classes]  (time-major, like CRNN output)
        input_lengths: [B]
        Returns: list of decoded strings (length B).
        """
        # Move to CPU
        if hasattr(log_probs, 'detach'):
            log_probs = log_probs.detach().cpu()
        T, B, _ = log_probs.shape
        results = []
        for b in range(B):
            seq_len = int(input_lengths[b]) if hasattr(input_lengths[b], 'item') else int(input_lengths[b])
            seq = log_probs[:seq_len, b, :]
            results.append(self.decode_single(seq))
        return results


# ============================================================
# Self-test
# ============================================================
def _self_test():
    import torch

    char_list = "abcdefghijklmnopqrstuvwxyz"
    vocab = ["cat", "car", "card", "care", "dog", "the", "thee"]
    decoder = WordBeamSearchDecoder(char_list, vocab, beam_width=10, mode="NGrams", lm_weight=0.5)

    # Synthesize log_probs that strongly suggest "card"
    T = 8
    num_classes = len(char_list) + 1
    BLANK = num_classes - 1
    log_probs = torch.full((T, 1, num_classes), -10.0)
    # Want: c (t=0), a (t=2), r (t=4), d (t=6) with blanks in between
    seq = [(0, 'c'), (2, 'a'), (4, 'r'), (6, 'd')]
    for t, ch in seq:
        log_probs[t, 0, char_list.index(ch)] = -0.1
    # Blanks at other positions
    for t in [1, 3, 5, 7]:
        log_probs[t, 0, BLANK] = -0.1

    # Normalize roughly
    log_probs = torch.log_softmax(log_probs, dim=-1)
    out = decoder.decode_batch(log_probs, [T])
    print(f"Test 1 (expect 'card'): got '{out[0]}'")
    assert out[0] == "card", f"Expected 'card', got '{out[0]}'"

    # Test with prefix that goes to a SHORTER word
    log_probs2 = torch.full((T, 1, num_classes), -10.0)
    seq2 = [(0, 'c'), (2, 'a'), (4, 't')]
    for t, ch in seq2:
        log_probs2[t, 0, char_list.index(ch)] = -0.1
    for t in [1, 3, 5, 6, 7]:
        log_probs2[t, 0, BLANK] = -0.1
    log_probs2 = torch.log_softmax(log_probs2, dim=-1)
    out2 = decoder.decode_batch(log_probs2, [T])
    print(f"Test 2 (expect 'cat'):  got '{out2[0]}'")
    assert out2[0] == "cat", f"Expected 'cat', got '{out2[0]}'"

    # Test OOV: try decoding "xyz" -> should fall back or return something
    log_probs3 = torch.full((T, 1, num_classes), -10.0)
    seq3 = [(0, 'x'), (2, 'y'), (4, 'z')]
    for t, ch in seq3:
        log_probs3[t, 0, char_list.index(ch)] = -0.1
    for t in [1, 3, 5, 6, 7]:
        log_probs3[t, 0, BLANK] = -0.1
    log_probs3 = torch.log_softmax(log_probs3, dim=-1)
    out3 = decoder.decode_batch(log_probs3, [T])
    print(f"Test 3 (no valid path, expect best vocab fallback): got '{out3[0]}'")
    # Should be a vocab word (closest valid path)
    assert out3[0] in vocab or out3[0] == "", f"Got unexpected '{out3[0]}'"

    print("All WBS self-tests PASSED")


if __name__ == "__main__":
    _self_test()
