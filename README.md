# CRNN Handwriting Recognition Project

Bu proje, el yazısı tanıma için CRNN (Convolutional Recurrent Neural Network) modelini kullanmaktadır.

## Özellikler

- **CRAFT** ile metin tespiti
- **CRNN** ile el yazısı tanıma
- **Trigram Language Model** ile hata düzeltme
- IAM veri seti üzerinde eğitim

## Dosya Yapısı

- `pipeline_v2.py` - Ana pipeline dosyası
- `greedy.py` - CRNN model eğitimi ve değerlendirmesi
- `trigram_lm.py` - Trigram dil modeli
- `CRAFT-pytorch-master/` - CRAFT metin tespit modeli
- `HTR_Models/` - Eğitilmiş modeller

## Kullanım

```bash
python pipeline_v2.py
```

## Gereksinimler

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- PIL

## Kurulum

```bash
pip install torch torchvision opencv-python numpy pillow
```

## Lisans

MIT License
