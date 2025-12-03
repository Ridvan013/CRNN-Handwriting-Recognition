#!/usr/bin/env python3
# -- coding: utf-8 --

"""
CRNN + CTC eğitimi (PyTorch versiyonu)
TensorFlow/Keras'tan PyTorch'a çevrilmiş versiyon
- CTC için metrics kaldırıldı (accuracy anlamsızdı)
- Adam(1e-3, clipnorm=5.0) + LR scheduler + EarlyStopping
- Zaman-adımı (T) dinamik ölçülür -> input_length otomatik
- Doğruluk yerine CER (Character Error Rate) callback ile raporlanır
"""

import os
import sys
import json
import math
import string
import subprocess
import time
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    print("PyTorch loaded successfully")
except ImportError as e:
    print(f"PyTorch import failed: {e}")
    print("Please install PyTorch: pip install torch torchvision")
    sys.exit(1)

# ==========================
# Basit paket garantörü
# ==========================
def ensure_package(pkg_name: str, import_name: str | None = None):
    import_name = import_name or pkg_name
    try:
        __import__(import_name)
    except Exception:
        print(f"[setup] installing: {pkg_name}")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=False)
        __import__(import_name)

# Gerekebilecek paketler
ensure_package("numpy")
ensure_package("opencv-python", "cv2")
ensure_package("matplotlib")
ensure_package("scikit-learn", "sklearn")
ensure_package("pandas")

# ==========================
# İçe aktarımlar
# ==========================
import numpy as np
import cv2

# Matplotlib backend ayarı (Tcl/Tk hatasını önlemek için)
import matplotlib
matplotlib.use('Agg')  # GUI olmayan backend kullan
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ==========================
# GPU Setup
# ==========================
def setup_gpu():
    """GPU kurulumu ve optimizasyonları"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU Bulundu: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # CuDNN optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Mixed Precision kontrolü
        if hasattr(torch.cuda, 'amp'):
            print("🎯 Mixed Precision (AMP) destekleniyor")
        else:
            print("⚠️ Mixed Precision desteklenmiyor")
            
        return device
    else:
        device = torch.device('cpu')
        print("⚠️ GPU bulunamadı, CPU kullanılıyor")
        return device

DEVICE = setup_gpu()

# ==========================
# Ortam bilgisi
# ==========================
IN_COLAB = False
try:
    import google.colab  # type: ignore
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally - Google Colab not available")

# ==========================
# Dataset yolları
# ==========================
# IAM words.txt satır biçimini kullanan veri yapısı
if IN_COLAB:
    WORDS_FILE = "words.txt"  # CRNN_1 klasöründe
    IMG_ROOT   = "words"      # CRNN_1/words/a/b/xxx.png gibi
else:
    # Script'in bulunduğu dizini bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Script directory: {script_dir}")
    
    # Olası dosya yollarını dene
    possible_words_files = [
        os.path.join(script_dir, "HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt"),
        os.path.join(script_dir, "HTR_Using_CRNN/IAM/processed/archive/words_new.txt"),
        os.path.join(script_dir, "words.txt"),
        "HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt",
        "HTR_Using_CRNN/IAM/processed/archive/words_new.txt",
        "words.txt"
    ]
    
    possible_img_roots = [
        os.path.join(script_dir, "HTR_Using_CRNN/IAM/processed/archive/iam_words/words"),
        "HTR_Using_CRNN/IAM/processed/archive/iam_words/words"
    ]
    
    # Words dosyasını bul
    WORDS_FILE = None
    for path in possible_words_files:
        if os.path.exists(path):
            WORDS_FILE = path
            break
    
    # Image root'u bul
    IMG_ROOT = None
    for path in possible_img_roots:
        if os.path.exists(path):
            IMG_ROOT = path
            break
    
    # Dosya yollarını kontrol et
    if WORDS_FILE is None:
        print(f"[ERROR] Words file not found in any of these locations:")
        for path in possible_words_files:
            print(f"  - {path}")
        print(f"Current directory: {os.getcwd()}")
        print("Please make sure the data files are in the correct location.")
        sys.exit(1)
    
    if IMG_ROOT is None:
        print(f"[ERROR] Image directory not found in any of these locations:")
        for path in possible_img_roots:
            print(f"  - {path}")
        print("Please check the image directory structure.")
        sys.exit(1)
    
    print(f"[INFO] Using words file: {WORDS_FILE}")
    print(f"[INFO] Using image root: {IMG_ROOT}")

# ==========================
# Karakter kümesi / yardımcılar
# ==========================
CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PAD_TOKEN = len(CHAR_LIST)  # pad için kullanıyoruz
BLANK_TOKEN = len(CHAR_LIST)  # CTC blank token

def encode_to_labels(txt: str) -> List[int]:
    out = []
    for ch in txt:
        if ch not in CHAR_LIST:
            raise ValueError(f"Unsupported char: {ch!r}")
        out.append(CHAR_LIST.index(ch))
    return out

def process_image_cpu_minimal(img_gray: np.ndarray) -> np.ndarray:
    """
    CPU'da minimal preprocessing - sadece dosya okuma ve temel format
    Tüm işlemler GPU'da yapılacak
    """
    if img_gray is None:
        raise ValueError("None image")
    
    # RGB'den grayscale'e çevir (eğer 3 kanallıysa) - CPU'da zorunlu
    if len(img_gray.shape) == 3 and img_gray.shape[2] == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    
    # Channel dimension ekle - GPU'da resize yapılacak
    if len(img_gray.shape) == 2:
        img_gray = np.expand_dims(img_gray, axis=-1)
    
    # Raw image'i döndür - GPU'da tüm işlemler yapılacak
    return img_gray

# ==========================
# Gelişmiş Augmentation Yardımcı Fonksiyonları
# ==========================
def elastic_transform(img: np.ndarray, alpha: float = 1000, sigma: float = 50) -> np.ndarray:
    """
    Elastic transform - el yazısı için mükemmel
    """
    try:
        h, w = img.shape[:2]
        
        # Random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)) * alpha
        dy = np.random.uniform(-1, 1, (h, w)) * alpha
        
        # Gaussian filter
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply transformation
        if len(img.shape) == 3:
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = cv2.remap(img[:, :, c], x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        else:
            result = cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return result
    except Exception:
        return img

def cutout(img: np.ndarray, mask_size: float = 0.3) -> np.ndarray:
    """
    Cutout - kısmi görünmezlik
    """
    try:
        h, w = img.shape[:2]
        mask_h = int(h * mask_size)
        mask_w = int(w * mask_size)
        
        # Random position
        y = np.random.randint(0, h - mask_h + 1)
        x = np.random.randint(0, w - mask_w + 1)
        
        # Apply cutout
        aug_img = img.copy()
        if len(aug_img.shape) == 3:
            aug_img[y:y+mask_h, x:x+mask_w, :] = 255
        else:
            aug_img[y:y+mask_h, x:x+mask_w] = 255
        
        return aug_img
    except Exception:
        return img

def random_erasing(img: np.ndarray, erasing_prob: float = 0.3) -> np.ndarray:
    """
    Random erasing - rastgele silme
    """
    try:
        if np.random.random() > erasing_prob:
            return img
            
        h, w = img.shape[:2]
        
        # Random erasing area
        area = np.random.uniform(0.02, 0.33) * h * w
        aspect_ratio = np.random.uniform(0.3, 3.0)
        
        mask_h = int(np.sqrt(area * aspect_ratio))
        mask_w = int(np.sqrt(area / aspect_ratio))
        
        if mask_h < h and mask_w < w:
            y = np.random.randint(0, h - mask_h)
            x = np.random.randint(0, w - mask_w)
            
            aug_img = img.copy()
            if len(aug_img.shape) == 3:
                aug_img[y:y+mask_h, x:x+mask_w, :] = np.random.uniform(0, 255, (mask_h, mask_w, aug_img.shape[2]))
            else:
                aug_img[y:y+mask_h, x:x+mask_w] = np.random.uniform(0, 255, (mask_h, mask_w))
            
            return aug_img
        return img
    except Exception:
        return img

def gaussian_noise(img: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """
    Gaussian noise - gelişmiş gürültü
    """
    try:
        noise = np.random.normal(0, noise_factor, img.shape)
        return np.clip(img + noise, 0, 1)
    except Exception:
        return img

def motion_blur(img: np.ndarray, max_kernel_size: int = 5) -> np.ndarray:
    """
    Motion blur - hareket bulanıklığı
    """
    try:
        if np.random.random() > 0.3:
            return img
            
        kernel_size = np.random.randint(3, max_kernel_size + 1)
        angle = np.random.uniform(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        # Apply blur
        if len(img.shape) == 3:
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
        else:
            result = cv2.filter2D(img, -1, kernel)
        
        return result
    except Exception:
        return img

def hybrid_augment_image(img: np.ndarray, augmentation_factor: int = 8) -> list:
    """
    Hibrit Data Augmentation: Mevcut + Gelişmiş teknikler
    El yazısı için optimize edilmiş augmentation
    """
    augmented_images = [img]  # Orijinal görüntüyü de dahil et
    
    for _ in range(augmentation_factor - 1):
        aug_img = img.copy()
        
        # MEVCUT TEKNİKLER (el yazısı için kanıtlanmış)
        
        # 1. Rotasyon (-5° ile +5° arası) - el yazısı için kritik
        if np.random.random() > 0.4:
            angle = np.random.uniform(-5, 5)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=255)
        
        # 2. Gelişmiş gürültü (daha kontrollü)
        if np.random.random() > 0.3:
            aug_img = gaussian_noise(aug_img, noise_factor=0.05)
        
        # 3. Kontrast değişimi (daha yumuşak)
        if np.random.random() > 0.4:
            contrast = np.random.uniform(0.85, 1.15)
            aug_img = np.clip(aug_img * contrast, 0, 1)
        
        # 4. Parlaklık değişimi (daha yumuşak)
        if np.random.random() > 0.4:
            brightness = np.random.uniform(-0.08, 0.08)
            aug_img = np.clip(aug_img + brightness, 0, 1)
        
        # 5. Hafif perspektif değişimi (daha kontrollü)
        if np.random.random() > 0.8:
            h, w = aug_img.shape[:2]
            offset = np.random.uniform(-0.5, 0.5, (4, 2))  # Daha küçük offset
            pts1 = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
            pts2 = pts1 + offset
            try:
                pts1_cv = np.array(pts1, dtype=np.float32)
                pts2_cv = np.array(pts2, dtype=np.float32)
                M = cv2.getPerspectiveTransform(pts1_cv, pts2_cv)
                aug_img = cv2.warpPerspective(aug_img, M, (w, h), borderValue=255)
            except Exception:
                # Fallback: hafif rotasyon
                angle = np.random.uniform(-1, 1)
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=255)
        
        # YENİ GELİŞMİŞ TEKNİKLER (el yazısı için optimize)
        
        # 6. Elastic Transform (el yazısı için mükemmel)
        if np.random.random() > 0.6:
            aug_img = elastic_transform(aug_img, alpha=500, sigma=30)  # Daha yumuşak
        
        # 7. Cutout (kısmi görünmezlik)
        if np.random.random() > 0.7:
            aug_img = cutout(aug_img, mask_size=0.2)  # Daha küçük cutout
        
        # 8. Random Erasing (mürekkep lekesi simülasyonu)
        if np.random.random() > 0.7:
            aug_img = random_erasing(aug_img, erasing_prob=0.2)
        
        # 9. Motion Blur (hareket bulanıklığı)
        if np.random.random() > 0.8:
            aug_img = motion_blur(aug_img, max_kernel_size=3)  # Hafif blur
        
        # 10. Gamma correction (daha doğal görünüm)
        if np.random.random() > 0.6:
            gamma = np.random.uniform(0.8, 1.2)
            aug_img = np.power(aug_img, gamma)
            aug_img = np.clip(aug_img, 0, 1)
        
        # Boyut kontrolü - tüm görüntüler aynı boyutta olmalı
        if aug_img.shape != img.shape:
            aug_img = cv2.resize(aug_img, (img.shape[1], img.shape[0]))
        
        # Final normalization
        aug_img = np.clip(aug_img, 0, 1)
        
        augmented_images.append(aug_img)
    
    return augmented_images

def augment_image(img: np.ndarray, augmentation_factor: int = 5) -> list:
    """
    Orijinal Data augmentation (geriye dönük uyumluluk için)
    """
    return hybrid_augment_image(img, augmentation_factor)

# ==========================
# PyTorch CRNN Model
# ==========================
class CRNNModel(nn.Module):
    def __init__(self, img_height: int = 32, img_width: int = 128, num_classes: int = None):
        super(CRNNModel, self).__init__()
        
        if num_classes is None:
            num_classes = len(CHAR_LIST) + 1  # +1 for blank token
        
        self.num_classes = num_classes
        
        # CNN Feature Extraction
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16,64)
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (8,32)
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),  # (4,32)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),  # (2,32)
            
            # Final conv
            nn.Conv2d(512, 512, kernel_size=2, padding=0),  # (1,31,512)
            nn.ReLU(inplace=True)
        )
        
        # RNN (Bidirectional LSTM)
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2),
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # CNN forward
        conv_features = self.cnn(x)  # [B, 512, 1, W']
        
        # Reshape for RNN: [B, C, H, W] -> [W, B, C] (time-major)
        B, C, H, W = conv_features.size()
        assert H == 1, f"Expected height=1, got {H}"
        
        # Squeeze height and permute for RNN
        rnn_input = conv_features.squeeze(2).permute(2, 0, 1)  # [W', B, C]
        
        # RNN forward
        lstm_out1, _ = self.rnn[0](rnn_input)
        lstm_out2, _ = self.rnn[1](lstm_out1)
        
        # Classifier
        logits = self.classifier(lstm_out2)  # [W', B, num_classes]
        
        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs

# ==========================
# GPU Data Augmentation Transforms (TAM GPU)
# ==========================
def get_gpu_transforms_full_gpu(is_training=True):
    """GPU'da yapılacak TAM data augmentation transforms - CPU'da hiçbir işlem yok"""
    if is_training:
        return transforms.Compose([
            # CPU'da sadece PIL'e çevir
            transforms.ToPILImage(),
            
            # GPU'da tüm augmentation'lar
            transforms.RandomRotation(degrees=5, fill=255),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=255),
            # transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # Hata veriyor, kaldırıldı
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))], p=0.2),  # Padding hatası, kaldırıldı
            
            # Tensor'e çevir
            transforms.ToTensor(),
            
            # GPU'da TAM preprocessing
            transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
            
            # GPU'da resize ve padding - TAM GPU işlemi
            transforms.Lambda(lambda x: F.interpolate(
                x.unsqueeze(0), 
                size=(32, 128), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)),
        ])
    else:
        return transforms.Compose([
            # CPU'da sadece PIL'e çevir
            transforms.ToPILImage(),
            transforms.ToTensor(),
            
            # GPU'da preprocessing
            transforms.Lambda(lambda x: 1.0 - x),  # Invert colors
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
            
            # GPU'da resize ve padding
            transforms.Lambda(lambda x: F.interpolate(
                x.unsqueeze(0), 
                size=(32, 128), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)),
        ])

# ==========================
# Custom Collate Function
# ==========================
def custom_collate_fn(batch):
    """Farklı boyutlardaki tensor'ları handle eder"""
    images, labels = zip(*batch)
    
    # Images'ları stack et (batch_size, channels, height, width)
    images = torch.stack(images, dim=0)
    
    # Labels'ları list olarak tut (CTC için gerekli)
    return images, list(labels)

# ==========================
# Dataset Class
# ==========================
class IAMDataset(Dataset):
    def __init__(self, images: List[np.ndarray], labels: List[List[int]], 
                 augmentation_factor: int = 1, is_training: bool = True, device='cuda'):
        self.images = images
        self.labels = labels
        self.augmentation_factor = augmentation_factor
        self.is_training = is_training
        self.device = device
        # Transform kaldırıldı - direkt GPU'da işlem yapılıyor
        
        # GPU Memory Pool - tüm tensors GPU'da kalacak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU Memory Pool temizlendi - TAM GPU Dataset için")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # CPU'da minimal işlem - sadece referans al
        image = self.images[idx]  # NumPy array (CPU'da)
        label = self.labels[idx]
        
        # Direkt GPU'ya transfer - PIL bypass (daha hızlı)
        # NumPy array'den direkt GPU tensor'e
        if len(image.shape) == 2:
            # Grayscale: (H, W) -> (1, H, W) format
            image = np.expand_dims(image, axis=0)  # (1, H, W)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # (H, W, 1) -> (1, H, W)
            image = np.transpose(image, (2, 0, 1))  # (1, H, W)
        elif len(image.shape) == 3:
            # (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        else:
            image = np.expand_dims(image, axis=0)
        
        # NumPy array'den CPU tensor'e, sonra GPU'ya transfer
        image_tensor = torch.from_numpy(image.copy()).float()  # CPU'da float tensor
        image_tensor = image_tensor.to(self.device, non_blocking=True)  # GPU'ya transfer
        
        # Normalize: [0, 255] -> [0, 1] (GPU'da)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # GPU'da TAM preprocessing ve augmentation
        if self.is_training:
            # Training: Augmentation GPU'da
            image_tensor = self._apply_gpu_augmentation(image_tensor)
        
        # GPU'da preprocessing (invert, normalize, resize)
        image_tensor = self._apply_gpu_preprocessing(image_tensor)
        
        # Label CPU'da kalacak - CTC için gerekli
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor
    
    def _apply_gpu_augmentation(self, img_tensor):
        """GPU'da augmentation uygula"""
        import random
        
        # Random rotation (GPU'da)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            img_tensor = TF.rotate(img_tensor.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
        
        # Random affine/translation (GPU'da)
        if random.random() < 0.5:
            # Translate: (dx, dy) in pixels
            h, w = img_tensor.shape[-2:]
            dx = random.uniform(-0.05, 0.05) * w
            dy = random.uniform(-0.05, 0.05) * h
            translate = [dx, dy]
            img_tensor = TF.affine(img_tensor.unsqueeze(0), angle=0, translate=translate, 
                                 scale=1.0, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR, fill=1.0).squeeze(0)
        
        # Color jitter (GPU'da)
        if random.random() < 0.5:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img_tensor = TF.adjust_brightness(img_tensor, brightness)
            img_tensor = TF.adjust_contrast(img_tensor, contrast)
        
        return img_tensor
    
    def _apply_gpu_preprocessing(self, img_tensor):
        """GPU'da preprocessing uygula"""
        # Invert colors (GPU'da)
        img_tensor = 1.0 - img_tensor
        
        # Normalize (GPU'da)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Resize to (32, 128) - GPU'da
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), 
            size=(32, 128), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return img_tensor

# ==========================
# CTC Loss Function
# ==========================
class CTCLoss(nn.Module):
    def __init__(self, blank_index: int = None):
        super(CTCLoss, self).__init__()
        if blank_index is None:
            blank_index = len(CHAR_LIST)
        self.blank_index = blank_index
        self.ctc_loss = nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

# ==========================
# Beam Search Decoding (GPU Optimized - Minimal CPU Usage)
# ==========================
def beam_search_decode(log_probs, input_lengths, beam_width=10, blank_index=None, device='cuda'):
    """
    GPU-optimized beam search decoding for CTC predictions
    Tüm tensor işlemleri GPU'da, sadece final sequence'lar CPU'ya alınıyor
    Args:
        log_probs: (sequence_length, batch_size, num_classes) - log probabilities from model
        input_lengths: (batch_size,) - actual sequence lengths
        beam_width: beam search width
        blank_index: blank token index
        device: device to run on
    Returns:
        List of decoded sequences (list of character indices)
    """
    if blank_index is None:
        blank_index = len(CHAR_LIST)  # Blank token index
    
    seq_len, batch_size, num_classes = log_probs.shape
    results = []
    
    # Ensure log_probs is on the correct device
    if log_probs.device != device:
        log_probs = log_probs.to(device)
    
    # Convert input_lengths to tensor if needed
    if isinstance(input_lengths, torch.Tensor):
        input_lengths = input_lengths.to(device)
    else:
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    
    for batch_idx in range(batch_size):
        seq_len_actual = int(input_lengths[batch_idx].item())  # Sadece bir kez CPU'ya al
        pred = log_probs[:seq_len_actual, batch_idx, :]  # (seq_len_actual, num_classes) - GPU'da
        
        # Initialize beam - log_prob GPU tensor olarak tut
        beam = [{'sequence': [], 'log_prob': torch.tensor(0.0, device=device, dtype=log_probs.dtype), 'last_char': None}]
        
        for t in range(seq_len_actual):
            new_beam = []
            
            # Get top-k characters for this timestep - GPU'da yap
            # pred[t] is (num_classes,) tensor on GPU
            top_k = min(beam_width * 2, num_classes)
            top_k_probs, top_k_indices = torch.topk(pred[t], top_k)  # GPU'da
            
            # Tüm top-k değerlerini GPU'da tut, sadece gerekli olduğunda CPU'ya al
            for beam_item in beam:
                # GPU'da log_prob hesapla
                expanded_log_probs = beam_item['log_prob'] + top_k_probs  # Broadcast: (beam_width,) + (top_k,)
                
                for k_idx in range(len(top_k_indices)):
                    # Sadece index için CPU'ya al (çok küçük veri)
                    char_idx = int(top_k_indices[k_idx].item())
                    char_log_prob = top_k_probs[k_idx]  # GPU tensor
                    
                    # Accumulate log probability (sum in log space) - GPU'da
                    new_log_prob = beam_item['log_prob'] + char_log_prob
                    
                    if char_idx == blank_index:
                        # Blank token - no change to sequence
                        new_beam.append({
                            'sequence': beam_item['sequence'].copy(),
                            'log_prob': new_log_prob,  # GPU tensor
                            'last_char': None
                        })
                    elif beam_item['last_char'] == char_idx:
                        # Same character as last - skip (CTC rule: no consecutive same chars)
                        new_beam.append({
                            'sequence': beam_item['sequence'].copy(),
                            'log_prob': new_log_prob,  # GPU tensor
                            'last_char': char_idx
                        })
                    else:
                        # New character - add to sequence
                        new_sequence = beam_item['sequence'].copy()
                        new_sequence.append(char_idx)
                        new_beam.append({
                            'sequence': new_sequence,
                            'log_prob': new_log_prob,  # GPU tensor
                            'last_char': char_idx
                        })
            
            # Keep only top beam_width items - GPU tensor'ları topla, sonra CPU'ya al
            # Log prob'ları GPU'da topla, sadece sorting için CPU'ya al
            if len(new_beam) > beam_width:
                # GPU'da log_prob değerlerini topla
                log_probs_tensor = torch.stack([item['log_prob'] for item in new_beam])  # GPU'da
                # Top beam_width'i al - GPU'da
                _, top_indices = torch.topk(log_probs_tensor, beam_width)
                # Sadece seçilenleri CPU'ya al
                beam = [new_beam[int(idx.item())] for idx in top_indices]
            else:
                beam = new_beam
        
        # Get best sequence - sadece final için CPU'ya al
        if beam:
            # GPU'da en iyi log_prob'u bul
            log_probs_final = torch.stack([item['log_prob'] for item in beam])
            best_idx = torch.argmax(log_probs_final).item()  # Sadece bir kez CPU'ya al
            best_sequence = beam[best_idx]['sequence']
        else:
            best_sequence = []
        results.append(best_sequence)
    
    return results

def greedy_decode(log_probs, input_lengths):
    """
    Greedy decoding for CTC predictions
    Model output: (sequence_length, batch_size, num_classes) - time-major format
    """
    seq_len, batch_size, num_classes = log_probs.shape
    results = []
    
    for batch_idx in range(batch_size):
        seq_len_actual = int(input_lengths[batch_idx])
        pred = log_probs[:seq_len_actual, batch_idx, :]  # (seq_len_actual, num_classes)
        
        # Greedy decode
        decoded = []
        prev_char = None
        
        for t in range(seq_len_actual):
            char_idx = torch.argmax(pred[t]).item()
            
            if char_idx != len(CHAR_LIST):  # Not blank
                if char_idx != prev_char:  # CTC rule: no consecutive same chars
                    decoded.append(char_idx)
                prev_char = char_idx
            else:
                prev_char = None
        
        results.append(decoded)
    
    return results

# ==========================
# Training Class
# ==========================
class CRNNTrainer:
    def __init__(self, model, device='auto', use_mixed_precision=True):
        self.model = model
        self.device = device if device != 'auto' else DEVICE
        self.model.to(self.device)
        
        # Mixed Precision
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("🎯 Mixed Precision (AMP) aktif")
        else:
            self.scaler = None
            print("⚠️ Mixed Precision kapalı")
        
        # GPU optimizasyonları
        if torch.cuda.is_available():
            # CuDNN optimizasyonları
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("🚀 CuDNN optimizasyonları aktif")
            
            # GPU Memory Pool optimizasyonları
            torch.cuda.empty_cache()  # Cache'i temizle
            torch.cuda.set_per_process_memory_fraction(0.9)  # GPU memory'nin %90'ını kullan
            print("🧹 GPU Memory Pool optimize edildi")
        
        # Loss function
        self.ctc_loss = CTCLoss(blank_index=len(CHAR_LIST))
        
        # Optimizer (TensorFlow Adam ile aynı parametreler)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=2, 
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cer': [],
            'val_wa': [],
            'val_wer': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_wa = 0.0
        self.best_epoch_loss = 0
        self.best_epoch_wa = 0
        self.patience_counter = 0
        self.early_stopping_patience = 8
        self.prev_val_loss = float('inf')  # Önceki epoch loss'u için
        
        # Input length cache - bir kez hesapla, sonra kullan (gereksiz forward pass'i önler)
        self.cached_input_length = None
        
    def _log_gpu_memory(self):
        """GPU memory kullanımını logla"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"🔋 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def train_epoch(self, train_loader, accumulation_steps=2, epoch_num=0):
        """Bir epoch eğitim - Gradient accumulation ile"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Data to device
            images = images.to(self.device, non_blocking=True)
            
            # Prepare CTC inputs
            batch_size = images.size(0)
            
            # Get input lengths (sequence length after CNN) - Cache'le, gereksiz forward pass yok
            if self.cached_input_length is None:
                with torch.no_grad():
                    dummy_output = self.model(images)
                    self.cached_input_length = dummy_output.size(0)
            input_lengths = torch.full((batch_size,), self.cached_input_length, dtype=torch.long, device=self.device)
            
            # Prepare targets - GPU'da direkt tensor olarak oluştur (CPU transfer yok)
            targets = []
            target_lengths = []
            for label in labels:
                # Label zaten CPU tensor, direkt extend et (tolist() kaldırıldı)
                if isinstance(label, torch.Tensor):
                    targets.extend(label.cpu().numpy().tolist())  # Sadece bir kez CPU'ya al
                else:
                    targets.extend(label)
                target_lengths.append(len(label) if isinstance(label, torch.Tensor) else len(label))
            
            # GPU'da direkt tensor oluştur
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    log_probs = self.model(images)
                    loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                log_probs = self.model(images)
                loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            num_batches += 1
            
            # Batch progress göster (her 20 batch'te bir veya son batch'te) - I/O azalt
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches
                progress_percent = int(progress * 100)
                bar_length = 20
                filled = int(progress * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
                avg_loss = total_loss / num_batches
                print(f"  Batch [{bar}] {progress_percent:3d}% | {batch_idx+1}/{total_batches} | Loss: {avg_loss:.4f}", end='\r')
            
            # GPU memory pool temizleme kaldırıldı - gereksiz overhead
            # PyTorch otomatik memory management yapıyor
        
        # Yeni satıra geç
        print()  # Progress bar'dan sonra yeni satır
        return total_loss / num_batches
    
    def evaluate(self, val_loader):
        """Validation değerlendirmesi"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Data to device
                images = images.to(self.device, non_blocking=True)
                
                # Prepare CTC inputs
                batch_size = images.size(0)
                
                # Prepare targets - GPU'da direkt tensor olarak oluştur (CPU transfer yok)
                targets = []
                target_lengths = []
                for label in labels:
                    # Label zaten CPU tensor, direkt extend et (tolist() kaldırıldı)
                    if isinstance(label, torch.Tensor):
                        targets.extend(label.cpu().numpy().tolist())  # Sadece bir kez CPU'ya al
                    else:
                        targets.extend(label)
                    target_lengths.append(len(label) if isinstance(label, torch.Tensor) else len(label))
                
                # GPU'da direkt tensor oluştur
                targets = torch.tensor(targets, dtype=torch.long, device=self.device)
                target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=self.device)
                
                # Forward pass - log_probs'tan sequence_length'i al (gereksiz dummy forward pass yok)
                if self.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        log_probs = self.model(images)
                        # Cache input length if not cached
                        if self.cached_input_length is None:
                            self.cached_input_length = log_probs.size(0)
                        input_lengths = torch.full((batch_size,), self.cached_input_length, dtype=torch.long, device=self.device)
                        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                else:
                    log_probs = self.model(images)
                    # Cache input length if not cached
                    if self.cached_input_length is None:
                        self.cached_input_length = log_probs.size(0)
                    input_lengths = torch.full((batch_size,), self.cached_input_length, dtype=torch.long, device=self.device)
                    loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Decode predictions for CER calculation - BEAM SEARCH kullan (GPU'da)
                predictions = beam_search_decode(log_probs, input_lengths, beam_width=10, 
                                                blank_index=len(CHAR_LIST), device=self.device)
                all_predictions.extend(predictions)
                all_targets.extend(labels)
        
        # Calculate CER, WA, WER
        cer, wa, wer = self._calculate_metrics(all_predictions, all_targets)
        
        return total_loss / num_batches, cer, wa, wer
    
    def _calculate_metrics(self, predictions, targets):
        """CER, WA, WER hesaplama"""
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        def indices_to_text(indices):
            return "".join(CHAR_LIST[i] for i in indices if i != -1)
        
        def dense_to_text(dense_row):
            return "".join(CHAR_LIST[i] for i in dense_row.tolist() if i != PAD_TOKEN)
        
        # Convert predictions and targets to text
        pred_texts = [indices_to_text(pred) for pred in predictions]
        true_texts = [dense_to_text(target) for target in targets]
        
        # Calculate CER
        cers = []
        for p, t in zip(pred_texts, true_texts):
            ed = levenshtein_distance(p, t)
            cers.append(ed / max(1, len(t)))
        cer_mean = float(np.mean(cers))
        
        # Calculate Word Accuracy
        word_correct = 0
        total_words = len(pred_texts)
        for p, t in zip(pred_texts, true_texts):
            if p.strip() == t.strip():
                word_correct += 1
        word_accuracy = word_correct / total_words if total_words > 0 else 0.0
        
        # Calculate Word Error Rate
        word_errors = total_words - word_correct
        wer = word_errors / total_words if total_words > 0 else 0.0
        
        return cer_mean, word_accuracy, wer
    
    def train(self, train_loader, val_loader, epochs=50):
        """Ana eğitim döngüsü"""
        print(f"\n{'=' * 63}")
        print("🚀 STARTING TRAINING WITH CRNN + CTC (PyTorch)")
        print(f"{'-' * 63}")
        print(f"  Training samples:   {len(train_loader.dataset):,}")
        print(f"  Validation samples:  {len(val_loader.dataset):,}")
        print(f"  Epochs:               {epochs}")
        print(f"  Device:               {self.device}")
        print(f"  Mixed Precision:     {self.use_mixed_precision}")
        print(f"  Beam Search Width:    10")
        print(f"{'=' * 63}\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training with gradient accumulation
            train_loss = self.train_epoch(train_loader, accumulation_steps=2, epoch_num=epoch)
            
            # Validation
            val_loss, val_cer, val_wa, val_wer = self.evaluate(val_loader)
            
            # Get current learning rate (scheduler.step'ten ÖNCE al - step sonrası değişebilir)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Learning rate scheduling (val_loss'a göre LR'ı azaltabilir)
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cer'].append(val_cer)
            self.history['val_wa'].append(val_wa)
            self.history['val_wer'].append(val_wer)
            self.history['learning_rate'].append(current_lr)
            
            # Get GPU memory info
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                gpu_memory_str = f"{allocated:.2f}GB / {total_memory:.2f}GB"
            else:
                gpu_memory_str = "N/A"
            
            # Check for best models
            is_best_loss = val_loss < self.best_val_loss
            is_best_wa = val_wa > self.best_val_wa
            
            # Loss improvement check
            loss_improved = val_loss < self.prev_val_loss
            loss_change = val_loss - self.prev_val_loss
            
            # Early stopping check
            if is_best_loss:
                self.best_val_loss = val_loss
                self.best_epoch_loss = epoch + 1
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_loss.pth')
            else:
                self.patience_counter += 1
            
            if is_best_wa:
                self.best_val_wa = val_wa
                self.best_epoch_wa = epoch + 1
                torch.save(self.model.state_dict(), 'best_model_wa.pth')
            
            # Print epoch results - Öneri 3 Format (Progress Bar + Kompakt)
            epoch_time = time.time() - start_time
            progress = (epoch + 1) / epochs
            progress_percent = int(progress * 100)
            bar_length = 20
            filled = int(progress * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Best model işaretleri
            best_loss_mark = " ⭐" if is_best_loss else ""
            best_wa_mark = " ⭐" if is_best_wa else ""
            
            # Loss improvement işareti
            if epoch > 0:  # İlk epoch'ta karşılaştırma yok
                if loss_improved:
                    loss_status = f"↓{abs(loss_change):.4f}"
                else:
                    loss_status = f"↑{abs(loss_change):.4f}"
            else:
                loss_status = ""
            
            # Best epoch bilgisi
            best_epoch_info = f" | Best Loss: E{self.best_epoch_loss}" if self.best_epoch_loss > 0 else ""
            best_wa_info = f" | Best WA: E{self.best_epoch_wa}" if self.best_epoch_wa > 0 else ""
            
            print(f"Epoch {epoch+1}/{epochs} [{bar}] {progress_percent:3d}% | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}{best_loss_mark} {loss_status} | "
                  f"CER: {val_cer:.4f} | WA: {val_wa:.4f}{best_wa_mark} | "
                  f"WER: {val_wer:.4f} | LR: {current_lr:.6f}{best_epoch_info}{best_wa_info} | {epoch_time:.1f}s")
            
            # Önceki loss'u güncelle
            self.prev_val_loss = val_loss
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch_loss}) | "
                      f"Best Val WA: {self.best_val_wa:.4f} (Epoch {self.best_epoch_wa})")
                break
        
        print(f"\n✅ Training completed! | Epochs: {len(self.history['train_loss'])} | "
              f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch_loss}) | "
              f"Best Val WA: {self.best_val_wa:.4f} (Epoch {self.best_epoch_wa})\n")
        return self.history

# ==========================
# Veriyi oku + Data Augmentation
# ==========================
RECORDS_COUNT = None  # Tüm IAM veri setini kullan (None = sınırsız, tüm veriyi yükle)
AUGMENTATION_FACTOR = 5  # Her görüntü için 5 farklı versiyon (hibrit augmentation için)
EPOCHS = 50  # Daha fazla veri için daha fazla epoch
BATCH_SIZE = 128  # Optimized batch size for better GPU utilization

def main():
    """Ana fonksiyon - Windows multiprocessing için gerekli"""
    global RECORDS_COUNT, AUGMENTATION_FACTOR, EPOCHS, BATCH_SIZE
    
    lines = []
if os.path.exists(WORDS_FILE):
    with open(WORDS_FILE, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    print(f"Loaded {len(lines)} lines from {WORDS_FILE}")
else:
    print(f"[ERROR] words file not found: {WORDS_FILE}")
    print(f"Available files in directory:")
    if os.path.exists("HTR_Using_CRNN/IAM/processed/archive"):
        for root, dirs, files in os.walk("HTR_Using_CRNN/IAM/processed/archive"):
            for file in files:
                if file.endswith('.txt'):
                    print(f"  - {os.path.join(root, file)}")
    sys.exit(1)

train_images, train_labels = [], []
valid_images, valid_labels = [], []

max_label_len = 0
processed_count = 0
skipped_invalid_format = 0
skipped_bad_status = 0
skipped_missing_image = 0
skipped_encode_error = 0

print(f"Starting data loading with {AUGMENTATION_FACTOR}x augmentation...")
print(f"Total lines in file: {len(lines):,}")

# Words klasöründeki gerçek resim sayısını kontrol et (sadece bilgi için)
if IMG_ROOT and os.path.exists(IMG_ROOT):
    actual_image_count = 0
    for root, dirs, files in os.walk(IMG_ROOT):
        actual_image_count += len([f for f in files if f.endswith('.png')])
    print(f"Total images in words folder: {actual_image_count:,} (for reference)")
else:
    print(f"⚠️  Image root not found: {IMG_ROOT}")
    actual_image_count = 0

# Sadece words.txt'deki satırlara göre veri yükle (GPU'da preprocessing yapılacak)
for idx, line in enumerate(lines):
    parts = line.split()
    if len(parts) < 9:
        skipped_invalid_format += 1
        continue
    status = parts[1]
    if status != "ok":
        skipped_bad_status += 1
        continue

    word_id = parts[0]
    word = "".join(parts[8:])

    # yol: words/a/a-b/xxx.png
    a, b, *_ = word_id.split('-')
    img_path = os.path.join(IMG_ROOT, a, f"{a}-{b}", f"{word_id}.png")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        skipped_missing_image += 1
        continue

    try:
        # CPU'da minimal preprocessing - sadece format kontrolü
        img = process_image_cpu_minimal(img)  # CPU'da minimal preprocessing
        lab = encode_to_labels(word)
    except Exception as e:
        skipped_encode_error += 1
        continue

    # Data augmentation uygula - SADECE GPU'DA YAPILACAK
    # Validation split: Her 10 görüntüden 1'i validation (9:1 split)
    # processed_count kullan (idx değil) - daha doğru split için
    # İlk görüntü training'e gitsin, sonra her 10'dan 1'i validation
    if processed_count > 0 and processed_count % 10 == 0:
        # Validation için augmentation yapma (sadece orijinal)
        valid_images.append(img)
        valid_labels.append(lab)
    else:
        # Training için sadece orijinal image'i ekle (augmentation GPU'da yapılacak)
        train_images.append(img)
        train_labels.append(lab)

    if len(word) > max_label_len:
        max_label_len = len(word)

    processed_count += 1
    if processed_count % 2000 == 0:
        print(f"Processed {processed_count:,} images, train: {len(train_images):,}, valid: {len(valid_images):,}")

    if RECORDS_COUNT is not None and processed_count >= RECORDS_COUNT:
        break

# Words klasöründeki toplam resim sayısını tekrar kontrol et (final)
if IMG_ROOT and os.path.exists(IMG_ROOT):
    actual_image_count = 0
    for root, dirs, files in os.walk(IMG_ROOT):
        actual_image_count += len([f for f in files if f.endswith('.png')])
else:
    actual_image_count = 0

print(f"\n{'=' * 63}")
print(f"Data loading completed!")
print(f"{'-' * 63}")
print(f"  Total lines in file:        {len(lines):,}")
print(f"  Total images in folder:     {actual_image_count:,} (for reference)")
print(f"  Successfully processed:     {processed_count:,}")
print(f"  Training images:            {len(train_images):,}")
print(f"  Validation images:          {len(valid_images):,}")
print(f"  Skipped (invalid format):   {skipped_invalid_format:,}")
print(f"  Skipped (bad status):       {skipped_bad_status:,}")
print(f"  Skipped (missing image):    {skipped_missing_image:,}")
print(f"  Skipped (encode error):     {skipped_encode_error:,}")
if actual_image_count > 0:
    coverage = (processed_count / actual_image_count) * 100
    print(f"  Coverage:                   {coverage:.1f}% ({processed_count:,}/{actual_image_count:,} images)")
print(f"  Augmentation factor:        {AUGMENTATION_FACTOR}x")
print(f"{'=' * 63}")

# Veri yükleme kontrolü
if len(train_images) == 0 and len(valid_images) == 0:
    print("\n[ERROR] No data loaded! Please check:")
    print("1. Words file path is correct")
    print("2. Image directory structure is correct")
    print("3. Image files are accessible")
    print(f"Words file: {WORDS_FILE}")
    print(f"Image root: {IMG_ROOT}")
    print("Exiting...")
    sys.exit(1)

# Veri hazır - GPU'da resize ve preprocessing yapılacak
print(f"\n{'=' * 63}")
print("📊 Dataset Summary")
print(f"{'-' * 63}")
print(f"  Training images:   {len(train_images):,}")
print(f"  Validation images: {len(valid_images):,}")
print(f"  Total images:      {len(train_images) + len(valid_images):,}")
print(f"  Images will be resized to (32, 128) on GPU during training")
print(f"{'=' * 63}\n")

# ==========================
# PyTorch DataLoader Setup
# ==========================
# Create datasets with GPU device
train_dataset = IAMDataset(train_images, train_labels, augmentation_factor=1, is_training=True, device=DEVICE)
val_dataset = IAMDataset(valid_images, valid_labels, augmentation_factor=1, is_training=False, device=DEVICE)

# Create data loaders with Windows-compatible settings
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,  # Windows multiprocessing sorunu için 0
    pin_memory=False,  # GPU tensor'ları için pin_memory gerekli değil
    drop_last=True,  # Son batch'i atla (GPU memory için)
    collate_fn=custom_collate_fn  # Custom collate function
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0,  # Windows multiprocessing sorunu için 0
    pin_memory=False,  # GPU tensor'ları için pin_memory gerekli değil
    drop_last=True,  # Son batch'i atla (GPU memory için)
    collate_fn=custom_collate_fn  # Custom collate function
)

print(f"DataLoader created: Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

# ==========================
# Model ve Eğitim
# ==========================
# Create model
model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST)+1)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Create trainer
trainer = CRNNTrainer(model, device='auto', use_mixed_precision=True)

# Start training
if len(train_images) > 0 and len(valid_images) > 0:
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
else:
    print("[WARN] Eğitim verisi yok, eğitim atlanıyor.")
    history = {'train_loss': [], 'val_loss': [], 'val_cer': [], 'val_wa': [], 'val_wer': []}

# ==========================
# Model Kaydetme
# ==========================
model_dir = "Model"
os.makedirs(model_dir, exist_ok=True)

# Save best models
best_loss_path = os.path.join(model_dir, "best_model_loss.pth")
best_wa_path = os.path.join(model_dir, "best_model_wa.pth")

if os.path.exists('best_model_loss.pth'):
    os.rename('best_model_loss.pth', best_loss_path)
    print(f"Best loss model saved to: {best_loss_path}")

if os.path.exists('best_model_wa.pth'):
    os.rename('best_model_wa.pth', best_wa_path)
    print(f"Best accuracy model saved to: {best_wa_path}")

# Save final model
final_model_path = os.path.join(model_dir, "Text_recognizer_Using_CRNN_PyTorch.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save training history
history_path = os.path.join(model_dir, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

# ==========================
# Detaylı Analiz ve Görselleştirme (GPU Optimized)
# ==========================
print(f"\n{'=' * 63}")
print("📊 Creating detailed analysis and visualizations...")
print(f"{'=' * 63}")

# Best model'i yükle (WA bazlı)
if os.path.exists(best_wa_path):
    model.load_state_dict(torch.load(best_wa_path, map_location=DEVICE))
    print(f"Loaded best WA model for analysis: {best_wa_path}")
    model.to(DEVICE)
    model.eval()

# Analiz fonksiyonlarını ekle
def levenshtein_distance(s1, s2):
    """Levenshtein distance hesaplar"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def create_detailed_analysis_csv(model, val_loader, model_dir, device='cuda'):
    """Test sonuçlarını detaylı analiz eden CSV dosyası oluşturur (GPU'da)"""
    try:
        import pandas as pd
        
        print("Creating detailed analysis CSV...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_images = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                batch_size = images.size(0)
                
                # Get predictions
                log_probs = model(images)
                sequence_length = log_probs.size(0)
                input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
                
                # Beam search decode (GPU'da)
                predictions = beam_search_decode(log_probs, input_lengths, beam_width=10, 
                                                blank_index=len(CHAR_LIST), device=device)
                
                all_predictions.extend(predictions)
                all_targets.extend(labels)
                
                # Images'ı CPU'ya al (visualization için)
                all_images.extend([img.cpu().numpy() for img in images])
        
        # Convert to text
        def indices_to_text(indices):
            return "".join(CHAR_LIST[i] for i in indices if i != -1)
        
        def dense_to_text(dense_row):
            if isinstance(dense_row, torch.Tensor):
                return "".join(CHAR_LIST[i] for i in dense_row.tolist() if i != PAD_TOKEN)
            else:
                return "".join(CHAR_LIST[i] for i in dense_row if i != PAD_TOKEN)
        
        pred_texts = [indices_to_text(pred) for pred in all_predictions]
        true_texts = [dense_to_text(target) for target in all_targets]
        
        # Analiz verilerini topla
        analysis_data = []
        
        for i, (pred, true) in enumerate(zip(pred_texts, true_texts)):
            is_correct = pred.strip() == true.strip()
            char_errors = levenshtein_distance(pred, true)
            word_length = len(true.strip())
            char_accuracy = 1 - (char_errors / max(1, len(true)))
            
            error_type = "None"
            if not is_correct:
                if len(pred) < len(true):
                    error_type = "Under-prediction"
                elif len(pred) > len(true):
                    error_type = "Over-prediction"
                else:
                    error_type = "Substitution"
            
            analysis_data.append({
                'Sample_ID': i,
                'True_Text': true,
                'Predicted_Text': pred,
                'Is_Correct': is_correct,
                'Word_Length': word_length,
                'Character_Errors': char_errors,
                'Character_Accuracy': round(char_accuracy, 4),
                'Error_Type': error_type,
                'Length_Difference': len(pred) - len(true)
            })
        
        # DataFrame oluştur
        df = pd.DataFrame(analysis_data)
        
        # CSV dosyasını kaydet
        csv_path = os.path.join(model_dir, "test_results_analysis.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Detailed analysis CSV saved to: {csv_path}")
        
        # Özet istatistikler
        create_summary_analysis(df, model_dir)
        
        return df, pred_texts, true_texts, all_images
        
    except Exception as e:
        print(f"❌ Error creating analysis CSV: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_summary_analysis(df, model_dir):
    """Özet analiz raporu oluşturur"""
    try:
        summary_path = os.path.join(model_dir, "test_summary_analysis.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== CRNN MODEL TEST RESULTS ANALYSIS ===\n\n")
            
            total_samples = len(df)
            correct_predictions = df['Is_Correct'].sum()
            accuracy = correct_predictions / total_samples
            
            f.write(f"TOTAL SAMPLES: {total_samples}\n")
            f.write(f"CORRECT PREDICTIONS: {correct_predictions}\n")
            f.write(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            # Kelime uzunluğuna göre analiz
            f.write("=== ANALYSIS BY WORD LENGTH ===\n")
            for length in sorted(df['Word_Length'].unique()):
                length_data = df[df['Word_Length'] == length]
                count = len(length_data)
                correct = length_data['Is_Correct'].sum()
                acc = correct / count
                char_acc = length_data['Character_Accuracy'].mean()
                
                f.write(f"Length {length}: {correct}/{count} correct ({acc:.4f}) - Char Acc: {char_acc:.4f}\n")
            
            f.write("\n=== ERROR TYPE ANALYSIS ===\n")
            error_analysis = df[df['Is_Correct'] == False]['Error_Type'].value_counts()
            for error_type, count in error_analysis.items():
                f.write(f"{error_type}: {count} cases\n")
            
            f.write("\n=== MOST ERROR-PRONE WORD LENGTHS ===\n")
            error_by_length = df[df['Is_Correct'] == False]['Word_Length'].value_counts().sort_values(ascending=False)
            for length, count in error_by_length.head(10).items():
                f.write(f"Length {length}: {count} errors\n")
            
            f.write("\n=== CHARACTER ACCURACY ANALYSIS ===\n")
            f.write(f"Mean Character Accuracy: {df['Character_Accuracy'].mean():.4f}\n")
            f.write(f"Min Character Accuracy: {df['Character_Accuracy'].min():.4f}\n")
            f.write(f"Max Character Accuracy: {df['Character_Accuracy'].max():.4f}\n")
            
            f.write("\n=== WORST PREDICTIONS ===\n")
            worst_predictions = df.nsmallest(5, 'Character_Accuracy')[['True_Text', 'Predicted_Text', 'Character_Accuracy']]
            for _, row in worst_predictions.iterrows():
                f.write(f"True: '{row['True_Text']}' | Pred: '{row['Predicted_Text']}' | Acc: {row['Character_Accuracy']:.4f}\n")
        
        print(f"✅ Summary analysis saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error creating summary analysis: {e}")

def visualize_predictions(pred_texts, true_texts, all_images, model_dir, num_examples=3):
    """Doğru ve yanlış tahminleri görselleştirir"""
    try:
        print("Creating prediction visualizations...")
        
        # Doğru ve yanlış tahminleri ayır
        correct_predictions = []
        incorrect_predictions = []
        
        for i, (pred, true) in enumerate(zip(pred_texts, true_texts)):
            if pred.strip() == true.strip():
                correct_predictions.append((i, pred, true, all_images[i]))
            else:
                incorrect_predictions.append((i, pred, true, all_images[i]))
        
        # Görselleştirme
        if len(correct_predictions) > 0 or len(incorrect_predictions) > 0:
            fig, axes = plt.subplots(2, num_examples, figsize=(4*num_examples, 8))
            fig.suptitle('📊 MODEL TAHMİN SONUÇLARI - Doğru vs Yanlış Örnekler', fontsize=16, fontweight='bold')
            
            # Doğru tahminler (üst satır)
            for idx in range(num_examples):
                if idx < len(correct_predictions):
                    img_idx, pred, true, img = correct_predictions[idx]
                    # Image preprocessing: (1, 32, 128) -> (32, 128)
                    if len(img.shape) == 3 and img.shape[0] == 1:
                        img = img.squeeze(0)
                    elif len(img.shape) == 4:
                        img = img.squeeze(0).squeeze(0)
                    axes[0, idx].imshow(img, cmap='gray')
                    axes[0, idx].set_title(f'✅ DOĞRU\nTrue: "{true}"\nPred: "{pred}"', 
                                         fontsize=12, fontweight='bold', color='green')
                else:
                    axes[0, idx].text(0.5, 0.5, 'Doğru örnek yok', ha='center', va='center', 
                                     transform=axes[0, idx].transAxes, fontsize=10, color='gray')
                axes[0, idx].axis('off')
            
            # Yanlış tahminler (alt satır)
            for idx in range(num_examples):
                if idx < len(incorrect_predictions):
                    img_idx, pred, true, img = incorrect_predictions[idx]
                    # Image preprocessing
                    if len(img.shape) == 3 and img.shape[0] == 1:
                        img = img.squeeze(0)
                    elif len(img.shape) == 4:
                        img = img.squeeze(0).squeeze(0)
                    axes[1, idx].imshow(img, cmap='gray')
                    axes[1, idx].set_title(f'❌ YANLIŞ\nTrue: "{true}"\nPred: "{pred}"', 
                                         fontsize=12, fontweight='bold', color='red')
                else:
                    axes[1, idx].text(0.5, 0.5, 'Yanlış örnek yok', ha='center', va='center', 
                                     transform=axes[1, idx].transAxes, fontsize=10, color='gray')
                axes[1, idx].axis('off')
            
            # Alt başlık ekle
            total = len(pred_texts)
            correct_count = len(correct_predictions)
            accuracy = correct_count / total * 100 if total > 0 else 0
            fig.text(0.5, 0.02, f'Toplam: {total} | Doğru: {correct_count} | Yanlış: {total-correct_count} | Accuracy: {accuracy:.1f}%', 
                    ha='center', fontsize=12, style='italic')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            
            # PNG dosyası olarak kaydet
            single_png_path = os.path.join(model_dir, "model_predictions_examples.png")
            plt.savefig(single_png_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Model predictions visualization saved to: {single_png_path}")
            plt.close()
        
        print(f"📊 Visualization Statistics:")
        print(f"   Total predictions: {len(pred_texts)}")
        print(f"   Correct predictions: {len(correct_predictions)} ({len(correct_predictions)/len(pred_texts)*100:.2f}%)")
        print(f"   Incorrect predictions: {len(incorrect_predictions)} ({len(incorrect_predictions)/len(pred_texts)*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error creating prediction visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_lr_comparison_plot(pytorch_history, model_dir, tensorflow_lr_config=None):
    """TensorFlow ve PyTorch learning rate'lerini karşılaştırır"""
    try:
        print("Creating learning rate comparison plot...")
        
        if 'learning_rate' not in pytorch_history or len(pytorch_history['learning_rate']) == 0:
            print("⚠️  No PyTorch learning rate data available for comparison")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Learning Rate Comparison: TensorFlow vs PyTorch', fontsize=16, fontweight='bold')
        
        epochs = range(len(pytorch_history['learning_rate']))
        pytorch_lr = pytorch_history['learning_rate']
        
        # TensorFlow LR schedule'ı simüle et (eğer config varsa)
        # TensorFlow: initial_lr=5e-4, ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)
        tensorflow_initial_lr = 5e-4
        tensorflow_factor = 0.3
        tensorflow_patience = 2
        tensorflow_min_lr = 1e-6
        
        # PyTorch LR schedule'ı (gerçek veri)
        axes[0].plot(epochs, pytorch_lr, label='PyTorch Learning Rate', 
                    color='blue', linewidth=2.5, marker='o', markersize=5)
        
        # TensorFlow LR schedule'ı simüle et (PyTorch validation loss'una göre)
        if 'val_loss' in pytorch_history and len(pytorch_history['val_loss']) > 0:
            tensorflow_lr_schedule = []
            current_tf_lr = tensorflow_initial_lr
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch_idx, val_loss in enumerate(pytorch_history['val_loss']):
                tensorflow_lr_schedule.append(current_tf_lr)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= tensorflow_patience:
                        new_lr = max(current_tf_lr * tensorflow_factor, tensorflow_min_lr)
                        if new_lr < current_tf_lr:
                            current_tf_lr = new_lr
                            patience_counter = 0
            
            axes[0].plot(epochs, tensorflow_lr_schedule, label='TensorFlow Learning Rate (Simulated)', 
                        color='red', linewidth=2.5, marker='s', markersize=5, linestyle='--', alpha=0.7)
        
        axes[0].set_title('Learning Rate Schedule Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Learning Rate', fontsize=12)
        axes[0].set_yscale('log')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, which='both')
        
        # LR değişim noktalarını işaretle
        # PyTorch LR değişimleri
        pytorch_changes = []
        for i in range(1, len(pytorch_lr)):
            if pytorch_lr[i] != pytorch_lr[i-1]:
                pytorch_changes.append(i)
                axes[0].axvline(x=i, color='blue', linestyle=':', alpha=0.5, linewidth=1)
        
        # TensorFlow LR değişimleri (simüle)
        if 'val_loss' in pytorch_history and len(pytorch_history['val_loss']) > 0:
            tensorflow_changes = []
            current_tf_lr = tensorflow_initial_lr
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch_idx, val_loss in enumerate(pytorch_history['val_loss']):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= tensorflow_patience:
                        new_lr = max(current_tf_lr * tensorflow_factor, tensorflow_min_lr)
                        if new_lr < current_tf_lr:
                            tensorflow_changes.append(epoch_idx)
                            current_tf_lr = new_lr
                            patience_counter = 0
            
            for change_epoch in tensorflow_changes:
                axes[0].axvline(x=change_epoch, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # İkinci grafik: LR farkı
        if 'val_loss' in pytorch_history and len(pytorch_history['val_loss']) > 0:
            lr_difference = []
            current_tf_lr = tensorflow_initial_lr
            patience_counter = 0
            best_val_loss = float('inf')
            
            for epoch_idx, val_loss in enumerate(pytorch_history['val_loss']):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= tensorflow_patience:
                        new_lr = max(current_tf_lr * tensorflow_factor, tensorflow_min_lr)
                        if new_lr < current_tf_lr:
                            current_tf_lr = new_lr
                            patience_counter = 0
                
                lr_diff = pytorch_lr[epoch_idx] - current_tf_lr
                lr_difference.append(lr_diff)
            
            axes[1].plot(epochs, lr_difference, label='LR Difference (PyTorch - TensorFlow)', 
                        color='purple', linewidth=2, marker='o', markersize=4)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            axes[1].set_title('Learning Rate Difference', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('LR Difference', fontsize=12)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
        
        # Bilgi kutusu ekle
        info_text = (
            f"TensorFlow Config:\n"
            f"  Initial LR: {tensorflow_initial_lr}\n"
            f"  Scheduler: ReduceLROnPlateau\n"
            f"  Factor: {tensorflow_factor}\n"
            f"  Patience: {tensorflow_patience}\n"
            f"  Min LR: {tensorflow_min_lr}\n\n"
            f"PyTorch Config:\n"
            f"  Initial LR: {pytorch_lr[0] if len(pytorch_lr) > 0 else 'N/A'}\n"
            f"  Scheduler: ReduceLROnPlateau\n"
            f"  Factor: 0.3\n"
            f"  Patience: 2\n"
            f"  Min LR: 1e-6"
        )
        fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # PNG dosyası olarak kaydet
        comparison_path = os.path.join(model_dir, "learning_rate_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Learning rate comparison saved to: {comparison_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating LR comparison plot: {e}")
        import traceback
        traceback.print_exc()

def create_training_plots(history, model_dir):
    """Eğitim sonuçlarını gösteren grafikleri oluşturur ve kaydeder"""
    try:
        print("Creating training plots...")
        
        # 3x2 layout (6 grafik için)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('CRNN Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Loss Grafiği
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Validation Accuracy (WA) Grafiği
        if 'val_wa' in history and len(history['val_wa']) > 0:
            axes[0, 1].plot(history['val_wa'], label='Validation Word Accuracy', color='green', linewidth=2)
            axes[0, 1].set_title('Validation Word Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Word Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation accuracy data', ha='center', va='center', 
                          transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Validation Word Accuracy', fontsize=14, fontweight='bold')
        
        # 3. CER (Character Error Rate) Grafiği
        if 'val_cer' in history and len(history['val_cer']) > 0:
            axes[1, 0].plot(history['val_cer'], label='Validation CER', color='orange', linewidth=2)
            axes[1, 0].set_title('Validation Character Error Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('CER')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No CER data', ha='center', va='center', 
                          transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Validation Character Error Rate', fontsize=14, fontweight='bold')
        
        # 4. WER (Word Error Rate) Grafiği
        if 'val_wer' in history and len(history['val_wer']) > 0:
            axes[1, 1].plot(history['val_wer'], label='Validation WER', color='purple', linewidth=2)
            axes[1, 1].set_title('Validation Word Error Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('WER')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No WER data', ha='center', va='center', 
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Validation Word Error Rate', fontsize=14, fontweight='bold')
        
        # 5. Learning Rate Grafiği (YENİ)
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            axes[2, 0].plot(history['learning_rate'], label='Learning Rate', color='darkblue', linewidth=2, marker='o', markersize=4)
            axes[2, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_yscale('log')  # Log scale çünkü LR genelde küçük değerler
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3, which='both')
            
            # LR değişim noktalarını işaretle
            lr_values = history['learning_rate']
            if len(lr_values) > 1:
                lr_changes = []
                for i in range(1, len(lr_values)):
                    if lr_values[i] != lr_values[i-1]:
                        lr_changes.append(i)
                
                if lr_changes:
                    for change_epoch in lr_changes:
                        axes[2, 0].axvline(x=change_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
                        axes[2, 0].text(change_epoch, lr_values[change_epoch] * 1.5, 
                                       f'LR↓\nE{change_epoch+1}', 
                                       ha='center', va='bottom', fontsize=8, color='red')
        else:
            axes[2, 0].text(0.5, 0.5, 'No learning rate data', ha='center', va='center', 
                          transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # 6. Learning Rate vs Loss (Karşılaştırma)
        if 'learning_rate' in history and 'val_loss' in history and len(history['learning_rate']) > 0 and len(history['val_loss']) > 0:
            ax_twin = axes[2, 1]
            ax_twin2 = ax_twin.twinx()
            
            epochs = range(len(history['val_loss']))
            line1 = ax_twin.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            line2 = ax_twin2.plot(epochs, history['learning_rate'], label='Learning Rate', color='darkblue', linewidth=2, marker='o', markersize=4)
            
            ax_twin.set_xlabel('Epoch', fontsize=12)
            ax_twin.set_ylabel('Validation Loss', color='red', fontsize=12)
            ax_twin2.set_ylabel('Learning Rate', color='darkblue', fontsize=12)
            ax_twin2.set_yscale('log')
            
            ax_twin.tick_params(axis='y', labelcolor='red')
            ax_twin2.tick_params(axis='y', labelcolor='darkblue')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_twin.legend(lines, labels, loc='upper right')
            
            ax_twin.set_title('Learning Rate vs Validation Loss', fontsize=14, fontweight='bold')
            ax_twin.grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No data for comparison', ha='center', va='center', 
                          transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Learning Rate vs Validation Loss', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Tek PNG olarak kaydet
        plot_path = os.path.join(model_dir, "training_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ All training plots saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

# Analiz ve görselleştirme işlemlerini çalıştır
if len(valid_images) > 0 and len(train_images) > 0:
    # CSV analizini oluştur
    analysis_df, pred_texts, true_texts, all_images = create_detailed_analysis_csv(
        model, val_loader, model_dir, device=DEVICE
    )
    
    # Tahmin görselleştirmelerini oluştur
    if pred_texts is not None and true_texts is not None and all_images is not None:
        visualize_predictions(pred_texts, true_texts, all_images, model_dir, num_examples=3)
    
    # Training grafiklerini oluştur
    if history and len(history.get('train_loss', [])) > 0:
        create_training_plots(history, model_dir)
else:
    print("⚠️  No validation data available for analysis")

print("DONE.")

if __name__ == '__main__':
    main()