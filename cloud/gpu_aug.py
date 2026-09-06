"""
Batched, per-sample-random augmentation on the GPU.

Why this exists
---------------
The original pipeline (``_augment_v4`` in v3_augmented_train.py) augments one
image at a time, on the CPU, with a separate resampling pass per transform.
With a 48k-word training set this leaves the GPU idle >90% of the time and
epochs take 10-50 minutes on the local RTX 4070.

This module applies the *same* transforms, with the *same* probabilities and
parameter ranges, to a whole batch at once on the GPU.  Every sample still
draws its own random parameters and its own "apply / don't apply" decision.

Differences from the per-image implementation (all deliberate)
--------------------------------------------------------------
* Rotation, translation, scale and shear are composed into ONE affine matrix
  and applied with a single bilinear resampling instead of four sequential
  ones.  Statistically equivalent; slightly less interpolation blur.
* The elastic displacement field is added to the same sampling grid, so the
  geometric + elastic stage is a single ``grid_sample`` call.
* Images are augmented at a fixed 64x256 intermediate resolution (chosen to
  match the median IAM word height) and then resized to the 32x128 model
  input, instead of being augmented at native resolution.  All five ablation
  configurations use this same path, so the ablation stays internally
  consistent.
* Random numbers come from torch's generator instead of Python's ``random``
  and NumPy.  ``torch.manual_seed`` therefore controls the augmentation stream.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Intermediate working resolution (H, W).  See module docstring.
AUG_H, AUG_W = 64, 256
# Model input resolution (H, W).
OUT_H, OUT_W = 32, 128

# (use_elastic, use_morph, use_wide_photometric)  -- mirrors _aug_flags()
MODE_FLAGS = {
    "full":    (True,  True,  True),
    "elastic": (True,  False, True),
    "morph":   (False, True,  True),
    "photo":   (False, False, True),
    "narrow":  (False, False, False),
}

# Probabilities and ranges, copied verbatim from _augment_v4.
P_ROTATE, ROT_DEG = 0.6, 7.0
P_TRANSLATE, TRANS_FRAC = 0.6, 0.05
P_SCALE, SCALE_LO, SCALE_HI = 0.3, 0.9, 1.1
P_SHEAR, SHEAR_DEG = 0.3, 5.0
P_ELASTIC, ELASTIC_ALPHA_LO, ELASTIC_ALPHA_HI, ELASTIC_SIGMA = 0.5, 2.0, 5.0, 0.08
P_MORPH = 0.3            # then k in {1,2} (k=1 is a no-op) and erode/dilate 50/50
P_PHOTO = 0.6
P_NOISE = 0.4
P_GAMMA = 0.4
P_ERASE = 0.3

PHOTO_WIDE, PHOTO_NARROW = (0.70, 1.35), (0.85, 1.15)
GAMMA_WIDE, GAMMA_NARROW = (0.70, 1.30), (0.80, 1.20)
NOISE_WIDE, NOISE_NARROW = 0.05, 0.03


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _bern(n: int, p: float, device) -> torch.Tensor:
    return torch.rand(n, device=device) < p


def _unif(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def _gaussian_kernel_1d(sigma: float, device) -> torch.Tensor:
    r = int(math.ceil(3.0 * sigma))
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


# Elastic amplitude parametrisation -- see configure_elastic().
ELASTIC_ALPHA = (ELASTIC_ALPHA_LO, ELASTIC_ALPHA_HI)
ELASTIC_LEGACY_AMPLITUDE = True


def configure_elastic(alpha_lo: float, alpha_hi: float, legacy_amplitude: bool) -> None:
    """Set how strong the elastic deformation is.

    legacy_amplitude=True  : alpha multiplies the *normalised* Gaussian-blurred
        noise exactly as ``_elastic_deform`` did.  Because a normalised blur of
        unit noise has an RMS of only ~0.01-0.02, alpha in [2,5] yields
        displacements of about 0.05-0.1 px RMS -- i.e. the original transform
        is almost a no-op.  Kept for faithful reproduction.
    legacy_amplitude=False : the blurred field is rescaled to unit RMS per
        sample, so alpha is the RMS displacement *in pixels*.  alpha in [1,3]
        is comparable to Simard et al. (2003) at this text height.
    """
    global ELASTIC_ALPHA, ELASTIC_LEGACY_AMPLITUDE
    ELASTIC_ALPHA = (float(alpha_lo), float(alpha_hi))
    ELASTIC_LEGACY_AMPLITUDE = bool(legacy_amplitude)


def _blurred_noise(n: int, H: int, W: int, sigma: float, device) -> torch.Tensor:
    """[n,1,H,W] Gaussian-smoothed U(-1,1) noise with uniform statistics.

    The noise is drawn on a canvas enlarged by the kernel radius and filtered
    with *valid* convolutions, so there is no padding artefact: with a radius
    comparable to the image height, replicate/reflect padding would inflate
    the field near the top and bottom edges.
    """
    k = _gaussian_kernel_1d(sigma, device)
    r = k.numel() // 2
    raw = torch.rand(n, 1, H + 2 * r, W + 2 * r, device=device) * 2.0 - 1.0
    f = F.conv2d(raw, k.view(1, 1, 1, -1))
    f = F.conv2d(f, k.view(1, 1, -1, 1))
    return f


def _bcast(mask: torch.Tensor) -> torch.Tensor:
    """[B] -> [B,1,1,1] for broadcasting against images."""
    return mask.view(-1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# geometric + elastic  (one grid_sample)
# --------------------------------------------------------------------------- #
def _affine_theta(B: int, H: int, W: int, device,
                  p_rot=P_ROTATE, p_trans=P_TRANSLATE,
                  p_scale=P_SCALE, p_shear=P_SHEAR) -> torch.Tensor:
    """Per-sample 2x3 sampling matrices in normalised coordinates.

    The forward image transform is  Shear ∘ Scale ∘ Translate ∘ Rotate  about
    the image centre (the order the per-image code applied them).
    ``grid_sample`` needs the *inverse* map (output -> input), so we build the
    forward 3x3 in pixel-centred coordinates, invert it, and convert to the
    [-1,1] normalised frame that ``affine_grid`` expects.
    """
    on_rot = _bern(B, p_rot, device)
    on_trans = _bern(B, p_trans, device)
    on_scale = _bern(B, p_scale, device)
    on_shear = _bern(B, p_shear, device)

    ang = torch.where(on_rot, _unif(B, -ROT_DEG, ROT_DEG, device), torch.zeros(B, device=device))
    ang = ang * math.pi / 180.0
    tx = torch.where(on_trans, _unif(B, -TRANS_FRAC, TRANS_FRAC, device) * W, torch.zeros(B, device=device))
    ty = torch.where(on_trans, _unif(B, -TRANS_FRAC, TRANS_FRAC, device) * H, torch.zeros(B, device=device))
    sc = torch.where(on_scale, _unif(B, SCALE_LO, SCALE_HI, device), torch.ones(B, device=device))
    sh = torch.where(on_shear, _unif(B, -SHEAR_DEG, SHEAR_DEG, device), torch.zeros(B, device=device))
    sh = torch.tan(sh * math.pi / 180.0)

    c, s = torch.cos(ang), torch.sin(ang)
    zero, one = torch.zeros(B, device=device), torch.ones(B, device=device)

    def m3(a, b, cc, d, e, f):  # rows [a b cc],[d e f],[0 0 1]
        return torch.stack([
            torch.stack([a, b, cc], 1),
            torch.stack([d, e, f], 1),
            torch.stack([zero, zero, one], 1),
        ], 1)

    R = m3(c, -s, zero, s, c, zero)
    T = m3(one, zero, tx, zero, one, ty)
    S = m3(sc, zero, zero, zero, sc, zero)
    Sh = m3(one, sh, zero, zero, one, zero)
    fwd = Sh @ S @ T @ R                     # [B,3,3] pixel-centred
    inv = torch.linalg.inv(fwd)

    # normalised frame: pixel = N · norm,  N = diag(W/2, H/2)
    n = torch.tensor([W / 2.0, H / 2.0], device=device)
    A = inv[:, :2, :2] * (n.view(1, 1, 2) / n.view(1, 2, 1))   # N^-1 A N
    t = inv[:, :2, 2] / n.view(1, 2)                           # N^-1 t
    return torch.cat([A, t.unsqueeze(-1)], dim=2)              # [B,2,3]


def _elastic_displacement(B: int, H: int, W: int, device,
                          p=P_ELASTIC) -> torch.Tensor:
    """Per-sample smooth displacement field in normalised units, [B,H,W,2]."""
    on = _bern(B, p, device)
    lo, hi = ELASTIC_ALPHA
    alpha = torch.where(on, _unif(B, lo, hi, device), torch.zeros(B, device=device))
    sigma_px = ELASTIC_SIGMA * max(H, W)
    disp = _blurred_noise(B * 2, H, W, sigma_px, device).view(B, 2, H, W)
    if not ELASTIC_LEGACY_AMPLITUDE:
        # unit-RMS field per sample -> alpha is the RMS displacement in pixels
        rms = disp.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        disp = disp / rms
    disp = disp * alpha.view(B, 1, 1, 1)                        # pixels
    # pixels -> normalised ([-1,1] spans W or H pixels)
    disp = torch.stack([disp[:, 0] * (2.0 / W), disp[:, 1] * (2.0 / H)], dim=-1)
    return disp                                                # [B,H,W,2]


def _warp(x: torch.Tensor, theta: torch.Tensor,
          disp: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply affine (+ optional elastic) with white padding.

    Background is 1.0 (white) and ink is dark, so we warp the *inverted*
    image with zero padding and invert back: out-of-image = white.
    """
    B, _, H, W = x.shape
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    if disp is not None:
        grid = grid + disp
    inv = F.grid_sample(1.0 - x, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return 1.0 - inv


# --------------------------------------------------------------------------- #
# morphology, photometric, noise, gamma, erasing
# --------------------------------------------------------------------------- #
def _morph(x: torch.Tensor, p=P_MORPH) -> torch.Tensor:
    B = x.shape[0]
    apply = _bern(B, p, x.device)
    k_is_2 = _bern(B, 0.5, x.device)          # k=1 would be a no-op
    use_erode = _bern(B, 0.5, x.device)
    m_ero = apply & k_is_2 & use_erode
    m_dil = apply & k_is_2 & ~use_erode
    if not (m_ero.any() or m_dil.any()):
        return x
    xp = F.pad(x, (0, 1, 0, 1), mode="replicate")
    dil = F.max_pool2d(xp, 2, stride=1)          # max filter: thins dark ink
    ero = -F.max_pool2d(-xp, 2, stride=1)        # min filter: thickens ink
    x = torch.where(_bcast(m_ero), ero, x)
    x = torch.where(_bcast(m_dil), dil, x)
    return x


def _photometric(x: torch.Tensor, wide: bool, p=P_PHOTO) -> torch.Tensor:
    B = x.shape[0]
    lo, hi = PHOTO_WIDE if wide else PHOTO_NARROW
    on = _bern(B, p, x.device)
    b = torch.where(on, _unif(B, lo, hi, x.device), torch.ones(B, device=x.device))
    c = torch.where(on, _unif(B, lo, hi, x.device), torch.ones(B, device=x.device))
    # torchvision adjust_brightness: clamp(img * b)
    x = (x * _bcast(b)).clamp_(0.0, 1.0)
    # torchvision adjust_contrast: clamp(c*img + (1-c)*mean(img))
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (_bcast(c) * x + (1.0 - _bcast(c)) * mean).clamp_(0.0, 1.0)
    return x


def _noise(x: torch.Tensor, wide: bool, p=P_NOISE) -> torch.Tensor:
    B = x.shape[0]
    sd = NOISE_WIDE if wide else NOISE_NARROW
    on = _bern(B, p, x.device).float()
    return (x + torch.randn_like(x) * sd * _bcast(on)).clamp_(0.0, 1.0)


def _gamma(x: torch.Tensor, wide: bool, p=P_GAMMA) -> torch.Tensor:
    B = x.shape[0]
    lo, hi = GAMMA_WIDE if wide else GAMMA_NARROW
    on = _bern(B, p, x.device)
    g = torch.where(on, _unif(B, lo, hi, x.device), torch.ones(B, device=x.device))
    return x.clamp(0.0, 1.0).pow(_bcast(g))


def _erase(x: torch.Tensor, p=P_ERASE) -> torch.Tensor:
    B, _, H, W = x.shape
    dev = x.device
    on = _bern(B, p, dev)
    ph_lo, ph_hi = max(1, H // 16), max(2, H // 6)
    pw_lo, pw_hi = max(1, W // 16), max(2, W // 6)
    ph = torch.randint(ph_lo, ph_hi + 1, (B,), device=dev)
    pw = torch.randint(pw_lo, pw_hi + 1, (B,), device=dev)
    y0 = (torch.rand(B, device=dev) * (H - ph + 1).float()).long()
    x0 = (torch.rand(B, device=dev) * (W - pw + 1).float()).long()
    rows = torch.arange(H, device=dev).view(1, H, 1)
    cols = torch.arange(W, device=dev).view(1, 1, W)
    m = ((rows >= y0.view(B, 1, 1)) & (rows < (y0 + ph).view(B, 1, 1)) &
         (cols >= x0.view(B, 1, 1)) & (cols < (x0 + pw).view(B, 1, 1)) &
         on.view(B, 1, 1))
    return torch.where(m.unsqueeze(1), torch.ones_like(x), x)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def augment_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Augment a batch [B,1,H,W] of float images in [0,1] (bg=1, ink dark).

    Transform order matches ``_augment_v4``:
    rotate/translate/scale/shear -> elastic -> morphology ->
    brightness+contrast -> noise -> gamma -> random erasing.
    """
    use_elastic, use_morph, wide = MODE_FLAGS[mode]
    B, _, H, W = x.shape
    theta = _affine_theta(B, H, W, x.device)
    disp = _elastic_displacement(B, H, W, x.device) if use_elastic else None
    x = _warp(x, theta, disp)
    if use_morph:
        x = _morph(x)
    x = _photometric(x, wide)
    x = _noise(x, wide)
    x = _gamma(x, wide)
    x = _erase(x)
    return x


def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched twin of model_v3._gpu_preprocess: invert, normalise, resize."""
    x = 1.0 - x
    x = (x - 0.5) / 0.5
    return F.interpolate(x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)


def resize_to_aug_res(img_u8: np.ndarray) -> np.ndarray:
    """Resize a grayscale uint8 image to the (AUG_H, AUG_W) working size."""
    import cv2
    if img_u8.shape[0] == AUG_H and img_u8.shape[1] == AUG_W:
        return img_u8
    return cv2.resize(img_u8, (AUG_W, AUG_H), interpolation=cv2.INTER_LINEAR)


class _LenOnly:
    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n


class GPUBatchLoader:
    """Drop-in replacement for the training/eval DataLoaders.

    Holds the whole split as one uint8 tensor on the GPU (train: ~786 MB) and
    yields ``(images, labels)`` where ``images`` is already the augmented,
    preprocessed [B,1,32,128] float batch on the device and ``labels`` is a
    list of 1-D LongTensors, exactly what ``CRNNTrainer`` and the evaluation
    helpers consume.  Exposes ``__len__`` and ``.dataset`` like a DataLoader.
    """

    def __init__(self, images: Sequence[np.ndarray], labels: Sequence[Sequence[int]],
                 batch_size: int, shuffle: bool, aug_mode: Optional[str],
                 device, drop_last: bool = False):
        arr = np.stack([resize_to_aug_res(im) for im in images])      # [N,H,W] uint8
        self.data = torch.from_numpy(arr).unsqueeze(1).to(device)      # uint8 on GPU
        self.labels: List[torch.Tensor] = [
            l if isinstance(l, torch.Tensor) else torch.as_tensor(list(l), dtype=torch.long)
            for l in labels
        ]
        self.N = self.data.shape[0]
        self.bs = batch_size
        self.shuffle = shuffle
        self.aug_mode = aug_mode
        self.device = device
        self.drop_last = drop_last
        self.dataset = _LenOnly(self.N)

    def __len__(self):
        return self.N // self.bs if self.drop_last else math.ceil(self.N / self.bs)

    def __iter__(self):
        idx = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        for b in range(len(self)):
            bi = idx[b * self.bs:(b + 1) * self.bs]
            x = self.data[bi.to(self.device)].float().div_(255.0)
            if self.aug_mode is not None:
                x = augment_batch(x, self.aug_mode)
            x = preprocess_batch(x)
            yield x, [self.labels[i] for i in bi.tolist()]
