"""
V3 CRNN model — importable module.

Kaynak: greedy_aachen_v3.py (feature/aachen-v3-extended-trigram branch)
Değişiklikler:
  - main() ve data loading kaldırıldı
  - CRNNTrainer: lr, warmup_epochs, total_epochs, patience parametrize edildi
  - freeze_cnn() / unfreeze_cnn() metotları eklendi
  - model_dir ctor parametresi yapıldı
"""

import os
import sys
import json
import math
import time
import warnings
from typing import List, Optional, Tuple
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2

matplotlib_ok = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib_ok = True
except ImportError:
    pass


# ─────────────────────────── GPU setup ──────────────────────────────────────

def setup_gpu() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    device = torch.device("cpu")
    print("WARNING: no CUDA GPU found, using CPU")
    return device

DEVICE = setup_gpu()


# ─────────────────────── Character set ──────────────────────────────────────

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PAD_TOKEN = len(CHAR_LIST)
BLANK_TOKEN = len(CHAR_LIST)   # CTC blank


def encode_to_labels(txt: str) -> List[int]:
    out = []
    for ch in txt:
        if ch not in CHAR_LIST:
            raise ValueError(f"Unsupported char: {ch!r}")
        out.append(CHAR_LIST.index(ch))
    return out


def decode_labels(indices: List[int]) -> str:
    return "".join(CHAR_LIST[i] for i in indices if i != BLANK_TOKEN)


def process_image_cpu_minimal(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None:
        raise ValueError("None image")
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    return img_gray.astype(np.uint8)


# ─────────────────────── CRNN Model (V3) ────────────────────────────────────

class CRNNModel(nn.Module):
    """
    V3 architecture: same CNN as V1/V2 + 4-layer BiLSTM hidden=512.
    Input: (B, 1, 32, 128)  →  Output (log-probs): (T, B, num_classes)
    """

    def __init__(self, img_height: int = 32, img_width: int = 128, num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = len(CHAR_LIST) + 1
        self.num_classes = num_classes
        self.hidden = 512

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Final
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512, hidden_size=self.hidden, num_layers=4,
            bidirectional=True, batch_first=False, dropout=0.3,
        )
        self.classifier = nn.Linear(2 * self.hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x)                           # (B, 512, 1, W')
        B, C, H, W = conv.size()
        assert H == 1, f"Expected H=1 after CNN, got {H}"
        rnn_in = conv.squeeze(2).permute(2, 0, 1)   # (W', B, 512)
        lstm_out, _ = self.rnn(rnn_in)               # (W', B, 2*hidden)
        logits = self.classifier(lstm_out)           # (W', B, num_classes)
        return F.log_softmax(logits, dim=2)


# ─────────────────────── Dataset ─────────────────────────────────────────────

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), list(labels)


class IAMDataset(Dataset):
    """GPU-augmented dataset for IAM word images (numpy arrays, grayscale)."""

    def __init__(self, images: List[np.ndarray], labels: List[List[int]],
                 is_training: bool = True, device: torch.device = None):
        self.images = images
        self.labels = labels
        self.is_training = is_training
        self.device = device or DEVICE

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # (H, W) → (1, H, W) tensor on GPU
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        t = torch.from_numpy(img.copy()).float().to(self.device, non_blocking=True)
        if t.max() > 1.0:
            t = t / 255.0

        if self.is_training:
            t = _gpu_augment(t)
        t = _gpu_preprocess(t)

        return t, torch.tensor(label, dtype=torch.long)


class SyntheticDataset(Dataset):
    """Dataset for trdg-generated synthetic word images."""

    def __init__(self, labels_file: str, img_dir: str,
                 is_training: bool = True, device: torch.device = None):
        self.img_dir = img_dir
        self.is_training = is_training
        self.device = device or DEVICE
        self.samples: List[Tuple[str, List[int]]] = []

        skipped = 0
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                fname, word = parts
                try:
                    lab = encode_to_labels(word)
                except ValueError:
                    skipped += 1
                    continue
                if len(lab) == 0 or len(lab) > 25:
                    skipped += 1
                    continue
                self.samples.append((fname, lab))
        if skipped:
            print(f"  SyntheticDataset: skipped {skipped} samples (unsupported chars / length)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank image on missing file
            img = np.ones((32, 128), dtype=np.uint8) * 255

        img_t = torch.from_numpy(img.copy()).float().unsqueeze(0)
        img_t = img_t.to(self.device, non_blocking=True)
        if img_t.max() > 1.0:
            img_t = img_t / 255.0

        if self.is_training:
            img_t = _gpu_augment(img_t)
        img_t = _gpu_preprocess(img_t)

        return img_t, torch.tensor(label, dtype=torch.long)


def _gpu_augment(img: torch.Tensor) -> torch.Tensor:
    import random
    if random.random() < 0.6:
        angle = random.uniform(-7, 7)
        img = TF.rotate(img.unsqueeze(0), angle,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.6:
        h, w = img.shape[-2:]
        dx = random.uniform(-0.05, 0.05) * w
        dy = random.uniform(-0.05, 0.05) * h
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[dx, dy],
                        scale=1.0, shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[0, 0],
                        scale=scale, shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.3:
        shear_x = random.uniform(-5, 5)
        img = TF.affine(img.unsqueeze(0), angle=0, translate=[0, 0],
                        scale=1.0, shear=[shear_x, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
    if random.random() < 0.6:
        img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
    if random.random() < 0.4:
        img = torch.clamp(img + torch.randn_like(img) * 0.03, 0.0, 1.0)
    if random.random() < 0.4:
        img = TF.adjust_gamma(img, random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        h, w = img.shape[-2:]
        ph = random.randint(max(1, h // 16), max(2, h // 6))
        pw = random.randint(max(1, w // 16), max(2, w // 6))
        y = random.randint(0, max(0, h - ph))
        x = random.randint(0, max(0, w - pw))
        img[..., y:y + ph, x:x + pw] = 1.0
    return img


def _gpu_preprocess(img: torch.Tensor) -> torch.Tensor:
    img = 1.0 - img                                   # invert
    img = (img - 0.5) / 0.5                           # normalize
    img = F.interpolate(img.unsqueeze(0), size=(32, 128),
                        mode="bilinear", align_corners=False).squeeze(0)
    return img


# ─────────────────────── Loss ────────────────────────────────────────────────

class CTCLoss(nn.Module):
    def __init__(self, blank_index: int = None):
        super().__init__()
        self.blank_index = blank_index if blank_index is not None else BLANK_TOKEN
        self.ctc = nn.CTCLoss(blank=self.blank_index, reduction="mean", zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


# ─────────────────────── Decoding ────────────────────────────────────────────

def greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
    T, B, _ = log_probs.shape
    results = []
    for b in range(B):
        seq_len = int(input_lengths[b])
        pred = log_probs[:seq_len, b, :]
        decoded, prev = [], None
        for t in range(seq_len):
            c = int(torch.argmax(pred[t]).item())
            if c != BLANK_TOKEN:
                if c != prev:
                    decoded.append(c)
                prev = c
            else:
                prev = None
        results.append(decoded)
    return results


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        cur = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def calculate_metrics(predictions: List[List[int]], targets) -> Tuple[float, float, float]:
    """Returns (CER, WA, WER)."""
    pred_texts = [decode_labels(p) for p in predictions]
    true_texts = []
    for t in targets:
        if isinstance(t, torch.Tensor):
            true_texts.append("".join(CHAR_LIST[i] for i in t.tolist() if i != PAD_TOKEN))
        else:
            true_texts.append(decode_labels(t))

    cers, correct = [], 0
    for p, t in zip(pred_texts, true_texts):
        cers.append(_levenshtein(p, t) / max(1, len(t)))
        if p.strip() == t.strip():
            correct += 1
    n = len(pred_texts)
    return float(np.mean(cers)), correct / n, (n - correct) / n


# ─────────────────────── Trainer ─────────────────────────────────────────────

class CRNNTrainer:
    """
    Parametrize edilmiş trainer — Phase 2 (pretrain) ve Phase 3 (finetune) için.

    lr           : peak learning rate (cosine schedule'ın zirvesi)
    warmup_epochs: linear warmup uzunluğu
    total_epochs : toplam epoch (scheduler hesabı için)
    patience     : early stopping patience
    model_dir    : checkpoint'lerin kaydedileceği dizin
    trigram_lm   : validation sırasında correction için (None = kapalı)
    """

    def __init__(self, model: CRNNModel,
                 lr: float = 7e-4,
                 warmup_epochs: int = 5,
                 total_epochs: int = 60,
                 patience: int = 15,
                 model_dir: str = "checkpoints",
                 device: torch.device = None,
                 trigram_lm=None):
        self.model = model
        self.device = device or DEVICE
        self.model.to(self.device)
        self.model_dir = model_dir
        self.trigram_lm = trigram_lm

        os.makedirs(model_dir, exist_ok=True)

        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        else:
            self.scaler = None

        self.ctc_loss = CTCLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        _we = warmup_epochs
        _te = total_epochs

        def _lr_lambda(epoch):
            if epoch < _we:
                return (epoch + 1) / _we
            prog = (epoch - _we) / max(1, _te - _we)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * prog))

        self.scheduler = LambdaLR(self.optimizer, _lr_lambda)
        self.patience = patience

        self.history = {k: [] for k in
                        ["train_loss", "val_loss", "val_cer", "val_wa", "val_wer", "lr",
                         "epoch_time_s"]}
        self.best_val_loss = float("inf")
        self.best_val_wa = 0.0
        self.best_epoch_loss = 0
        self.best_epoch_wa = 0
        self._patience_counter = 0
        self._cached_T = None

    # ── CNN freeze / unfreeze (Phase 3 için) ──────────────────────────────────

    def freeze_cnn(self):
        for p in self.model.cnn.parameters():
            p.requires_grad = False
        print("  CNN frozen")

    def unfreeze_cnn(self):
        for p in self.model.cnn.parameters():
            p.requires_grad = True
        # Reset optimizer state so unfrozen params get fresh momentum
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=1e-5,
        )
        print("  CNN unfrozen, optimizer reset")

    # ── One training epoch ────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, accum: int = 2, epoch: int = 0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            B = images.size(0)

            if self._cached_T is None:
                with torch.no_grad():
                    self._cached_T = self.model(images).size(0)
            input_lengths = torch.full((B,), self._cached_T, dtype=torch.long, device=self.device)

            flat, tlens = [], []
            for lab in labels:
                lst = lab.cpu().numpy().tolist() if isinstance(lab, torch.Tensor) else list(lab)
                flat.extend(lst)
                tlens.append(len(lst))
            targets = torch.tensor(flat, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(tlens, dtype=torch.long, device=self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    lp = self.model(images)
                    loss = self.ctc_loss(lp, targets, input_lengths, target_lengths) / accum
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                lp = self.model(images)
                loss = self.ctc_loss(lp, targets, input_lengths, target_lengths) / accum
                loss.backward()
                if (batch_idx + 1) % accum == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * accum
            n += 1
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                pct = int((batch_idx + 1) / len(loader) * 100)
                print(f"  [{pct:3d}%] loss={total_loss/n:.4f}", end="\r")
        print()
        return total_loss / n

    # ── Validation ────────────────────────────────────────────────────────────

    def _evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                B = images.size(0)
                flat, tlens = [], []
                for lab in labels:
                    lst = lab.cpu().numpy().tolist() if isinstance(lab, torch.Tensor) else list(lab)
                    flat.extend(lst)
                    tlens.append(len(lst))
                targets = torch.tensor(flat, dtype=torch.long, device=self.device)
                target_lengths = torch.tensor(tlens, dtype=torch.long, device=self.device)

                ctx = torch.amp.autocast("cuda") if self.use_amp else torch.no_grad()
                with ctx:
                    lp = self.model(images)
                    if self._cached_T is None:
                        self._cached_T = lp.size(0)
                    input_lengths = torch.full((B,), self._cached_T, dtype=torch.long, device=self.device)
                    loss = self.ctc_loss(lp, targets, input_lengths, target_lengths)

                total_loss += loss.item()
                n += 1
                preds = greedy_decode(lp, input_lengths)

                if self.trigram_lm:
                    corrected = []
                    for p in preds:
                        txt = decode_labels(p)
                        txt_c = self.trigram_lm.correct_word(txt)
                        corrected.append([CHAR_LIST.index(c) for c in txt_c if c in CHAR_LIST])
                    all_preds.extend(corrected)
                else:
                    all_preds.extend(preds)
                all_targets.extend(labels)

        cer, wa, wer = calculate_metrics(all_preds, all_targets)
        return total_loss / n, cer, wa, wer

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 60, cnn_freeze_epochs: int = 0) -> dict:
        """
        cnn_freeze_epochs: CNN freeze kaç epoch sürecek (0 = freeze yok).
        Phase 3'te 5 verilir.
        """
        print(f"\n{'='*60}")
        print(f" CRNN Training   epochs={epochs}  device={self.device}")
        print(f" train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")
        print(f" CNN freeze first {cnn_freeze_epochs} epochs: {cnn_freeze_epochs > 0}")
        print(f"{'='*60}\n")

        if cnn_freeze_epochs > 0:
            self.freeze_cnn()

        for epoch in range(epochs):
            t0 = time.time()

            # Unfreeze CNN after freeze period
            if cnn_freeze_epochs > 0 and epoch == cnn_freeze_epochs:
                self.unfreeze_cnn()

            train_loss = self._train_epoch(train_loader, accum=2, epoch=epoch)
            val_loss, val_cer, val_wa, val_wer = self._evaluate(val_loader)
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            epoch_time = time.time() - t0
            for k, v in zip(
                ["train_loss", "val_loss", "val_cer", "val_wa", "val_wer", "lr", "epoch_time_s"],
                [train_loss, val_loss, val_cer, val_wa, val_wer, cur_lr, epoch_time],
            ):
                self.history[k].append(v)

            is_best_loss = val_loss < self.best_val_loss
            is_best_wa = val_wa > self.best_val_wa

            if is_best_loss:
                self.best_val_loss = val_loss
                self.best_epoch_loss = epoch + 1
                self._patience_counter = 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_dir, "best_model_loss.pth"))
            else:
                self._patience_counter += 1

            if is_best_wa:
                self.best_val_wa = val_wa
                self.best_epoch_wa = epoch + 1
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_dir, "best_model_wa.pth"))

            star_l = " *" if is_best_loss else ""
            star_w = " *" if is_best_wa else ""
            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}{star_l}  "
                f"CER={val_cer:.4f}  WA={val_wa:.4f}{star_w}  "
                f"lr={cur_lr:.2e}  {time.time()-t0:.1f}s"
            )

            if self._patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\nDone. Best WA={self.best_val_wa:.4f} (ep {self.best_epoch_wa}), "
              f"Best loss={self.best_val_loss:.4f} (ep {self.best_epoch_loss})")

        history_path = os.path.join(self.model_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved → {history_path}")

        return self.history


# ─────────────────────── Statistical eval ────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for proportion."""
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Tuple[float, float]:
    """
    Paired McNemar test. Returns (chi2_statistic, p_value).
    correct_a[i] = True if model A got sample i right.
    """
    from scipy.stats import chi2
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))
    c = sum(not a and bb for a, bb in zip(correct_a, correct_b))
    if b + c == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)   # continuity correction
    p = 1.0 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p
