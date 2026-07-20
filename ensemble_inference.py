"""
V1 + V2 + V3 Ensemble Inference on Aachen Test Set (5,338 samples).
Softmax averaging in log-space + V3 Extended Trigram correction.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import math
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from trigram_lm import TrigramLanguageModel

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHAR_LIST) + 1


# ============================================================
# V1 architecture (Sequential of 2 single-layer LSTMs, h=256)
# ============================================================
class CRNN_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2),
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2)
        )
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        conv = self.cnn(x)
        rnn_in = conv.squeeze(2).permute(2, 0, 1)
        lstm1, _ = self.rnn[0](rnn_in)
        lstm2, _ = self.rnn[1](lstm1)
        logits = self.classifier(lstm2)
        return F.log_softmax(logits, dim=2)


# ============================================================
# V2 architecture (3-layer LSTM, h=384)
# ============================================================
class CRNN_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 384
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=self.hidden, num_layers=3,
                          bidirectional=True, batch_first=False, dropout=0.2)
        self.classifier = nn.Linear(2 * self.hidden, NUM_CLASSES)

    def forward(self, x):
        conv = self.cnn(x)
        rnn_in = conv.squeeze(2).permute(2, 0, 1)
        lstm_out, _ = self.rnn(rnn_in)
        logits = self.classifier(lstm_out)
        return F.log_softmax(logits, dim=2)


# ============================================================
# V3 architecture (4-layer LSTM, h=512)
# ============================================================
class CRNN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 512
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=self.hidden, num_layers=4,
                          bidirectional=True, batch_first=False, dropout=0.3)
        self.classifier = nn.Linear(2 * self.hidden, NUM_CLASSES)

    def forward(self, x):
        conv = self.cnn(x)
        rnn_in = conv.squeeze(2).permute(2, 0, 1)
        lstm_out, _ = self.rnn(rnn_in)
        logits = self.classifier(lstm_out)
        return F.log_softmax(logits, dim=2)


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_t = torch.from_numpy(img).float() / 255.0
    img_t = 1.0 - img_t
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    img_t = F.interpolate(img_t, size=(32, 128), mode='bilinear', align_corners=False)
    return img_t.squeeze(0)


def greedy_decode_from_log_probs(log_probs):
    """Greedy CTC decoding. log_probs: [T, B, C]."""
    seq_len, B, C = log_probs.shape
    results = []
    blank_idx = len(CHAR_LIST)
    for b in range(B):
        decoded = []
        prev = None
        for t in range(seq_len):
            idx = torch.argmax(log_probs[t, b]).item()
            if idx != blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx
        text = "".join(CHAR_LIST[i] for i in decoded if 0 <= i < len(CHAR_LIST))
        results.append(text)
    return results


def load_test_samples(
    test_uttlist: str = None,
    img_root: str = None,
    words_txt: str = None,
):
    """Load Aachen test (ok-only, ~5,338 samples).

    Args can override; defaults auto-search common paths (Kaggle + local).
    """
    def _resolve(explicit, candidates):
        if explicit and os.path.exists(explicit):
            return explicit
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    test_uttlist = _resolve(test_uttlist, [
        str(REPO_ROOT / "aachen_splits" / "splits" / "test.uttlist"),
        "/kaggle/working/aachen_splits/splits/test.uttlist",
        "aachen_splits/splits/test.uttlist",
    ])
    img_root = _resolve(img_root, [
        str(REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" / "iam_words" / "words"),
        os.environ.get("IAM_ROOT", ""),
    ])
    words_txt = _resolve(words_txt, [
        str(REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" / "iam_words" / "words.txt"),
        os.environ.get("IAM_WORDS", ""),
    ])
    if not (test_uttlist and img_root and words_txt):
        raise FileNotFoundError(
            "Aachen paths bulunamadı. Set IAM_ROOT/IAM_WORDS env vars or pass args.\n"
            f"  test.uttlist : {test_uttlist}\n  img_root     : {img_root}\n  words.txt    : {words_txt}"
        )
    samples = []
    test_forms = set()
    with open(test_uttlist) as f:
        for line in f:
            line = line.strip()
            if line:
                test_forms.add(line)
    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            word_id = parts[0]
            status = parts[1]
            if status != "ok":
                continue
            form_id = "-".join(word_id.split("-")[:2])
            if form_id not in test_forms:
                continue
            transcription = parts[-1]
            a, b = word_id.split("-")[:2]
            img_path = os.path.join(img_root, a, f"{a}-{b}", f"{word_id}.png")
            if os.path.exists(img_path):
                samples.append((word_id, transcription, img_path))
    return samples


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center - margin), min(1, center + margin))


def mcnemar_test(preds_a, preds_b, targets, label_a, label_b):
    """Paired McNemar test."""
    a_correct = [p == t for p, t in zip(preds_a, targets)]
    b_correct = [p == t for p, t in zip(preds_b, targets)]
    b = sum(1 for i in range(len(targets)) if a_correct[i] and not b_correct[i])
    c = sum(1 for i in range(len(targets)) if not a_correct[i] and b_correct[i])
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_asymp = math.erfc(math.sqrt(chi2 / 2.0))
        n_disc = b + c
        k = min(b, c)
        p_exact = 2 * sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
        p_exact = min(1.0, p_exact)
    else:
        chi2 = 0.0
        p_asymp = 1.0
        p_exact = 1.0
    n_a = sum(a_correct)
    n_b = sum(b_correct)
    print(f"\n  McNemar: {label_a} vs {label_b}")
    print(f"    {label_a}: {n_a}/{len(targets)} = {n_a/len(targets)*100:.2f}%")
    print(f"    {label_b}: {n_b}/{len(targets)} = {n_b/len(targets)*100:.2f}%")
    print(f"    Delta: {(n_b-n_a)/len(targets)*100:+.2f}pp, chi2={chi2:.2f}, p_exact={p_exact:.3e}, b={b}, c={c}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading V1 (8.75M params, 2-layer LSTM h=256)...")
    m1 = CRNN_V1().to(device)
    m1.load_state_dict(torch.load(str(REPO_ROOT / "Model_aachen" / "best_model_wa.pth"),
                                  map_location=device))
    m1.eval()

    print("Loading V2 (15.46M params, 3-layer LSTM h=384)...")
    m2 = CRNN_V2().to(device)
    m2.load_state_dict(torch.load(str(REPO_ROOT / "Model_aachen_v2" / "best_model_wa.pth"),
                                  map_location=device))
    m2.eval()

    print("Loading V3 (28.73M params, 4-layer LSTM h=512)...")
    m3 = CRNN_V3().to(device)
    m3.load_state_dict(torch.load(str(REPO_ROOT / "Model_aachen_v3" / "best_model_wa.pth"),
                                  map_location=device))
    m3.eval()

    print("\nLoading V3 Extended Trigram (auto NLTK)...")
    lm = TrigramLanguageModel(str(REPO_ROOT / "aachen_splits" / "train_words.txt"))

    print("\nLoading Aachen test samples (ok-only)...")
    samples = load_test_samples()
    print(f"  {len(samples)} samples")

    # Storage for predictions
    v3_preds = []
    ens_preds_raw = []
    ens_preds_trigram = []
    targets = []

    BATCH = 64
    print("\nRunning ensemble inference...")
    with torch.no_grad():
        for i in range(0, len(samples), BATCH):
            batch = samples[i:i+BATCH]
            imgs = [preprocess_image(p) for _, _, p in batch]
            imgs = [im for im in imgs if im is not None]
            if not imgs:
                continue
            batch_tensor = torch.stack(imgs).to(device)

            # Get log_probs from each model
            lp1 = m1(batch_tensor)  # [T, B, C]
            lp2 = m2(batch_tensor)
            lp3 = m3(batch_tensor)

            # Ensemble: convert to softmax, average, back to log
            # (averaging log_probs directly is also valid; both work well)
            p1 = torch.exp(lp1)
            p2 = torch.exp(lp2)
            p3 = torch.exp(lp3)
            p_ens = (p1 + p2 + p3) / 3.0
            lp_ens = torch.log(p_ens + 1e-10)

            # Decode
            preds_v3 = greedy_decode_from_log_probs(lp3)
            preds_ens = greedy_decode_from_log_probs(lp_ens)

            for k, (_, true, _) in enumerate(batch):
                if k >= len(preds_ens):
                    continue
                v3_preds.append(preds_v3[k])
                ens_preds_raw.append(preds_ens[k])
                ens_preds_trigram.append(lm.correct_word(preds_ens[k]))
                targets.append(true)

            processed = min(i + BATCH, len(samples))
            if processed % 500 < BATCH or processed == len(samples):
                n_ok_v3 = sum(1 for p, t in zip(v3_preds, targets) if p == t)
                n_ok_ens = sum(1 for p, t in zip(ens_preds_trigram, targets) if p == t)
                print(f"  {processed}/{len(samples)} | V3: {n_ok_v3/len(targets)*100:.2f}% | "
                      f"Ensemble+Trigram: {n_ok_ens/len(targets)*100:.2f}%")

    N = len(targets)
    correct_v3 = sum(1 for p, t in zip(v3_preds, targets) if p == t)
    correct_ens_raw = sum(1 for p, t in zip(ens_preds_raw, targets) if p == t)
    correct_ens_tri = sum(1 for p, t in zip(ens_preds_trigram, targets) if p == t)

    print(f"\n{'='*72}")
    print(f"  ENSEMBLE V1+V2+V3 RESULTS - AACHEN TEST SET (ok-only, {N} samples)")
    print(f"{'='*72}")
    print(f"  V3 alone (greedy):              {correct_v3}/{N} = {correct_v3/N*100:.2f}%")
    print(f"  Ensemble (greedy, no LM):       {correct_ens_raw}/{N} = {correct_ens_raw/N*100:.2f}%")
    print(f"  Ensemble + V3 Extended Trigram: {correct_ens_tri}/{N} = {correct_ens_tri/N*100:.2f}%")
    ci = wilson_ci(correct_ens_tri, N)
    print(f"  Wilson 95% CI (Ensemble+Trigram): [{ci[0]*100:.2f}, {ci[1]*100:.2f}]")

    # McNemar tests
    mcnemar_test(v3_preds, ens_preds_raw, targets, "V3 alone", "Ensemble raw")
    mcnemar_test(v3_preds, ens_preds_trigram, targets, "V3 alone", "Ensemble + Trigram")

    # Save CSV
    csv_out = str(REPO_ROOT / "Model_ensemble" / "test_results_analysis.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "True_Text", "V3_Prediction", "Ensemble_Raw", "Ensemble_Trigram", "Is_Correct"])
        for i, (v3p, ep, et, t) in enumerate(zip(v3_preds, ens_preds_raw, ens_preds_trigram, targets)):
            writer.writerow([i, t, v3p, ep, et, et == t])
    print(f"\nCSV saved: {csv_out}")


if __name__ == "__main__":
    main()
