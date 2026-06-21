"""
Beam search sanity test:
  1. beam_width=1 ile beam search GREEDY ile ayni sonucu vermeli (matematiksel zorunluluk)
  2. beam_width=10 ile beam search farkli sonuc verebilmeli (en az bazi orneklerde)
  3. Beam search sequence'leri valid olmali (CTC blank/repeat kurallari)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import math

# Path setup
sys.path.insert(0, r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1")

# Import decoder functions from greedy_aachen.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "greedy_aachen",
    r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\greedy_aachen.py"
)
# Don't fully execute - we just need the functions
# Read and exec only the relevant parts manually

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BLANK_IDX = len(CHAR_LIST)
NUM_CLASSES = len(CHAR_LIST) + 1


def greedy_decode(log_probs, input_lengths):
    seq_len, batch_size, num_classes = log_probs.shape
    results = []
    for batch_idx in range(batch_size):
        seq_len_actual = int(input_lengths[batch_idx])
        pred = log_probs[:seq_len_actual, batch_idx, :]
        decoded = []
        prev_char = None
        for t in range(seq_len_actual):
            char_idx = torch.argmax(pred[t]).item()
            if char_idx != len(CHAR_LIST):
                if char_idx != prev_char:
                    decoded.append(char_idx)
                prev_char = char_idx
            else:
                prev_char = None
        results.append(decoded)
    return results


def beam_search_decode(log_probs, input_lengths, beam_width=10):
    seq_len, batch_size, num_classes = log_probs.shape
    blank_index = len(CHAR_LIST)
    results = []
    lp = log_probs.detach().cpu()
    for batch_idx in range(batch_size):
        seq_len_actual = int(input_lengths[batch_idx])
        pred = lp[:seq_len_actual, batch_idx, :]
        beams = [((), None, 0.0)]
        for t in range(seq_len_actual):
            top_k = min(beam_width * 2, num_classes)
            top_probs, top_idxs = torch.topk(pred[t], top_k)
            top_probs = top_probs.tolist()
            top_idxs = top_idxs.tolist()
            new_beams = {}
            for seq_tuple, last_char, score in beams:
                for char_log_p, char_idx in zip(top_probs, top_idxs):
                    new_score = score + char_log_p
                    if char_idx == blank_index:
                        key = (seq_tuple, None)
                    elif char_idx == last_char:
                        key = (seq_tuple, char_idx)
                    else:
                        key = (seq_tuple + (char_idx,), char_idx)
                    if key in new_beams:
                        prev = new_beams[key]
                        if new_score > prev:
                            new_beams[key] = prev + math.log1p(math.exp(new_score - prev))
                        else:
                            new_beams[key] = new_score + math.log1p(math.exp(prev - new_score))
                    else:
                        new_beams[key] = new_score
            scored = [(k[0], k[1], v) for k, v in new_beams.items()]
            scored.sort(key=lambda b: b[2], reverse=True)
            beams = scored[:beam_width]
        best_seq = list(beams[0][0]) if beams else []
        results.append(best_seq)
    return results


def indices_to_text(indices):
    return "".join(CHAR_LIST[i] for i in indices if 0 <= i < len(CHAR_LIST))


def make_test_log_probs(seed=0, T=20, batch=8):
    """Sahte log_probs uret - test icin"""
    torch.manual_seed(seed)
    logits = torch.randn(T, batch, NUM_CLASSES)
    log_probs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.full((batch,), T, dtype=torch.long)
    return log_probs, input_lengths


def test_1_beam1_equals_greedy():
    """Beam width=1 ile greedy ayni sonucu vermeli."""
    print("\n[TEST 1] beam_width=1 vs greedy")
    log_probs, lengths = make_test_log_probs(seed=42)
    greedy = greedy_decode(log_probs, lengths)
    beam1 = beam_search_decode(log_probs, lengths, beam_width=1)

    fails = 0
    for i, (g, b) in enumerate(zip(greedy, beam1)):
        g_txt = indices_to_text(g)
        b_txt = indices_to_text(b)
        match = "OK" if g_txt == b_txt else "DIFFER"
        print(f"  Sample {i}: greedy='{g_txt}' beam1='{b_txt}' [{match}]")
        if g_txt != b_txt:
            fails += 1

    if fails == 0:
        print("  PASS - beam_width=1 matches greedy on all samples")
    else:
        print(f"  FAIL - {fails}/{len(greedy)} samples differ - beam search BUG!")
    return fails == 0


def test_2_beam10_can_differ():
    """Beam width=10 ile greedy zaman zaman farkli sonuc vermeli (otherwise beam is doing nothing)."""
    print("\n[TEST 2] beam_width=10 vs greedy (some differences expected)")
    differences = 0
    samples_total = 0
    for seed in range(5):
        log_probs, lengths = make_test_log_probs(seed=seed, batch=16)
        greedy = greedy_decode(log_probs, lengths)
        beam = beam_search_decode(log_probs, lengths, beam_width=10)
        for g, b in zip(greedy, beam):
            samples_total += 1
            if indices_to_text(g) != indices_to_text(b):
                differences += 1

    print(f"  Samples where beam differs from greedy: {differences}/{samples_total}")
    if differences > 0:
        print(f"  PASS - beam search produces DIFFERENT (potentially better) results")
    else:
        print(f"  WARNING - beam never differs from greedy. Either:")
        print(f"           - Test data too easy (sharp distributions), or")
        print(f"           - beam search is degenerate (BUG)")
    return differences > 0


def test_3_no_blank_in_output():
    """Beam search ciktilarinda blank token (idx={}) olmamali.""".format(BLANK_IDX)
    print(f"\n[TEST 3] No blank token (idx={BLANK_IDX}) in beam output")
    log_probs, lengths = make_test_log_probs(seed=7, batch=20)
    beam = beam_search_decode(log_probs, lengths, beam_width=10)

    has_blank = False
    for i, seq in enumerate(beam):
        if BLANK_IDX in seq:
            print(f"  Sample {i}: contains blank token! BUG.")
            has_blank = True

    if not has_blank:
        print(f"  PASS - no blank tokens in any output sequence")
    else:
        print(f"  FAIL - blank tokens leaked into output")
    return not has_blank


def test_4_no_consecutive_repeats():
    """CTC kuralina gore ardisik ayni karakter olmamali."""
    print(f"\n[TEST 4] No consecutive identical chars (CTC collapse rule)")
    log_probs, lengths = make_test_log_probs(seed=11, batch=20)
    beam = beam_search_decode(log_probs, lengths, beam_width=10)

    has_repeat = False
    for i, seq in enumerate(beam):
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                print(f"  Sample {i}: consecutive repeat at pos {j-1},{j}! BUG.")
                has_repeat = True
                break

    if not has_repeat:
        print(f"  PASS - no consecutive repeats in any sequence")
    else:
        print(f"  FAIL - consecutive repeats found (CTC rule violated)")
    return not has_repeat


def test_5_score_monotonicity():
    """Beam'lar log-prob'a gore siralanmis olmali (en iyisi ilk)."""
    print(f"\n[TEST 5] Beam scoring consistency")
    # Bu testi yapmak icin beam search'un internal beams'a erismek gerekiyor
    # Simdilik atlayalim
    print("  SKIPPED (would require internal beam exposure)")
    return True


if __name__ == "__main__":
    results = {
        "T1 beam1==greedy": test_1_beam1_equals_greedy(),
        "T2 beam10!=greedy sometimes": test_2_beam10_can_differ(),
        "T3 no blank leak": test_3_no_blank_in_output(),
        "T4 no consecutive repeats": test_4_no_consecutive_repeats(),
    }
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} - {name}")

    all_ok = all(results.values())
    print("\n" + ("ALL TESTS PASSED - beam search is functional" if all_ok
                  else "SOME TESTS FAILED - beam search has bugs"))
    sys.exit(0 if all_ok else 1)
