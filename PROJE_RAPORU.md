# El Yazısı Tanıma Sistemi (HTR) - Proje Teknik Raporu

## 1. Proje Özeti
Bu proje, karmaşık el yazısı metin görüntülerini dijital metne dönüştürmek amacıyla geliştirilmiş uçtan uca (end-to-end) bir Derin Öğrenme sistemidir. Sistem, metin tespiti (Text Detection) ve metin tanıma (Text Recognition) olmak üzere iki ana aşamadan oluşmaktadır. IAM Handwriting Database üzerinde eğitilmiş ve test edilmiştir.

## 2. Sistem Mimarisi

Proje, modern OCR (Optik Karakter Tanıma) yaklaşımlarını takip eden hibrit bir mimari kullanır:

### 2.1. Metin Tespiti (Text Detection) - CRAFT
Görüntü üzerindeki metin bölgelerinin yerini belirlemek için **CRAFT (Character Region Awareness for Text Detection)** modeli kullanılmıştır.
- **Çalışma Prensibi:** Karakter bölgelerini ve karakterler arası bağlantıları (affinity) tahmin eder.
- **Avantajı:** Düzensiz, kavisli veya deforme olmuş metin satırlarını başarıyla tespit edebilir.
- **Çıktı:** Metin kutularının (bounding boxes) koordinatları.

### 2.2. Metin Tanıma (Text Recognition) - CRNN
Tespit edilen metin bölgelerinin içeriğini okumak için **CRNN (Convolutional Recurrent Neural Network)** mimarisi kullanılmıştır. Bu mimari üç ana bileşenden oluşur:

1.  **Konvolüsyonel Katmanlar (CNN):**
    - Görüntüden görsel öznitelikleri (features) çıkarır.
    - 7 katmanlı bir CNN yapısı kullanılmıştır.
    - Girdi görüntüleri 32x128 boyutuna normalize edilir.

2.  **Tekrarlayan Katmanlar (RNN - BiLSTM):**
    - CNN'den gelen öznitelik dizilerini işler.
    - **Bidirectional LSTM (BiLSTM)** kullanılarak, metnin hem geçmiş hem de gelecek bağlamı (context) öğrenilir.
    - Her biri 256 gizli birime (hidden units) sahip iki katmanlı LSTM yapısı vardır.

3.  **Transkripsiyon Katmanı (CTC Loss):**
    - **Connectionist Temporal Classification (CTC)**, RNN çıktısını karakter dizisine dönüştürür.
    - Karakter hizalaması (alignment) gerektirmeden eğitim yapılmasını sağlar.

### 2.3. Hata Düzeltme (Post-Processing)
- **Beam Search Decoding:** En olası karakter dizilerini bulmak için kullanılır.
- **Trigram Language Model:** Olasılık tabanlı dil modeli ile kelime hatalarını düzeltmek için entegre edilmiştir (Opsiyonel).

---

## 3. Teknik Gereksinimler ve Bağımlılıklar

Projenin çalıştırılması için aşağıdaki yazılım ve kütüphaneler gereklidir:

### 3.1. Yazılım Ortamı
- **Dil:** Python 3.8+
- **İşletim Sistemi:** Windows / Linux / macOS

### 3.2. Temel Kütüphaneler (`requirements.txt`)
Aşağıdaki komut ile tüm bağımlılıklar yüklenebilir:
```bash
pip install torch torchvision opencv-python numpy pillow scipy scikit-image
```

| Kütüphane | Sürüm (Önerilen) | Amaç |
|-----------|------------------|------|
| `torch` | 1.13+ | Derin öğrenme model altyapısı (NN, Autograd) |
| `torchvision` | 0.14+ | Görüntü dönüşümleri ve veri yükleme |
| `opencv-python`| 4.x | Görüntü okuma, işleme ve görselleştirme |
| `numpy` | 1.21+ | Matris ve dizi işlemleri |
| `Pillow` | 9.x | Görüntü formatı işlemleri |

### 3.3. Donanım Gereksinimleri
- **Önerilen:** NVIDIA GPU (CUDA destekli) - Model çıkarımı (inference) süresini 10-20 kat hızlandırır.
- **Minimum:** Modern bir CPU (Intel i5/i7 veya AMD Ryzen 5/7).

---

## 4. Dosya Yapısı ve Modüller

Proje aşağıdaki temel dosya ve klasörlerden oluşur:

*   **`pipeline_v2.py`**: Sistemin ana giriş noktasıdır. Görüntüyü alır, CRAFT ile metni bulur, CRNN ile okur ve sonucu görselleştirir.
*   **`greedy.py`**: CRNN modelinin eğitimi, veri yükleme (Dataset class) ve değerlendirme (Evaluation) kodlarını içerir.
*   **`trigram_lm.py`**: İstatistiksel dil modeli (Trigram) sınıfını içerir.
*   **`CRAFT-pytorch-master/`**: CRAFT modelinin kaynak kodları ve ağırlıkları.
*   **`Model/`**: Eğitilmiş model dosyaları (`.pth`).
    *   `best_model_wa.pth`: Kelime doğruluğuna (Word Accuracy) göre en iyi model.
    *   `best_model_loss.pth`: En düşük kayıp değerine (Loss) göre en iyi model.

---

## 5. Kurulum ve Kullanım

### 5.1. Kurulum
Projeyi klonlayın ve bağımlılıkları yükleyin:
```bash
git clone https://github.com/Ridvan013/CRNN-Handwriting-Recognition.git
cd CRNN-Handwriting-Recognition
pip install -r requirements.txt
```

### 5.2. Çalıştırma
Pipeline'ı bir görüntü üzerinde çalıştırmak için:

```bash
python pipeline_v2.py --image "resim_yolu.png" --cuda True
```

**Parametreler:**
- `--image`: İşlenecek görüntünün yolu.
- `--cuda`: GPU kullanımı (True/False).
- `--text_threshold`: Metin tespit hassasiyeti (Varsayılan: 0.7).

---

## 6. Performans ve Sonuçlar

Sistem, IAM veri seti üzerinde eğitilmiş olup, el yazısı cümleleri yüksek doğrulukla tanıyabilmektedir.

**Örnek Çıktı:**
- **Girdi:** El yazısı içeren bir görüntü.
- **İşlem:**
    1.  Metin blokları tespit edilir.
    2.  Bloklar satır satır ayrıştırılır ve sıralanır.
    3.  Her kelime CRNN modeline beslenir.
- **Çıktı:** Dijital metin dosyası (`.txt`) ve görselleştirilmiş sonuç (`.jpg`).

---

## 7. Gelecek Çalışmalar (Future Work)
- **Transformer Entegrasyonu:** RNN yerine Transformer tabanlı modellerin denenmesi.
- **Veri Çoğaltma (Augmentation):** Daha fazla el yazısı varyasyonu için Elastic Transform kullanımı.
- **Mobil Entegrasyon:** Modelin ONNX formatına çevrilerek mobil cihazlarda çalıştırılması.
