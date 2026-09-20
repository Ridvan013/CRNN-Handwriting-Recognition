# Ablation Sonuçları — tam IAM, Aachen writer-disjoint (7 Eylül 2026)

Tek makine (RTX 4070 Laptop, PyTorch 2.7.1+cu118, Python 3.13), tek kod
sürümü (`075688d`), beş konfigürasyon sıfırdan eğitildi. Değişen tek şey
`--aug-mode`; epoch=100, batch=128, lr=7e-4, patience=15, seed=42,
elastik genlik düzeltilmiş (`--elastic-legacy-amplitude 0 --elastic-alpha 1 3`).

Veri: train 47.997 / val 7.205 / test **20.310** kelime (`status=ok`,
resmi 747/116/336 form; val'den metni testte de geçen 5 form çıkarıldı;
yazar/metin/form ayrıklığı `verify_aachen_splits.py` ile doğrulandı, 19/19).

Test her mod için **en iyi doğrulama checkpoint'iyle** ölçüldü. Kanıt: aynı
checkpoint ayrı süreçte iki kez çalıştırıldığında 0 fark; farklı checkpoint
(`best_model_loss.pth`) 4.875 farklı tahmin, %73,7 doğruluk.

## Eğitim özeti

| Mod | epoch | en iyi val WA | @epoch | son val WA | durdurma |
|---|---:|---:|---:|---:|---|
| narrow (CRNN-B) | 100 | 84,61 | 87 | 84,58 | epoch sınırı |
| photo | 91 | 84,34 | 76 | 83,75 | 76+15 |
| elastic | 92 | 84,69 | 77 | 84,40 | 77+15 |
| morph | 99 | 84,58 | 84 | 84,40 | 84+15 |
| full (CRNN-LX) | 85 | 84,77 | 70 | 84,41 | 70+15 |

## Tablo A — augmentation ablation (test, N=20.310, greedy + IAM+NLTK trigram)

Nihai sayılar `results/ablation_lexicon_<mod>.json` (deterministik beraberlik
çözümü, `075688d`). Wilson %95 aralığı yaklaşık ±0,55 pp.

| Konfigürasyon | wide photo | elastic | morph | WA (%) | CER (%) | Δ vs CRNN-L |
|---|:---:|:---:|:---:|---:|---:|---:|
| CRNN-B (baseline) | ✗ | ✗ | ✗ | 80,35 | 9,34 | — |
| + wide photometric | ✓ | ✗ | ✗ | 80,34 | 9,34 | −0,01 |
| + elastic | ✓ | ✓ | ✗ | 80,68 | 9,09 | +0,33 |
| + morphological | ✓ | ✗ | ✓ | 80,64 | 9,09 | +0,29 |
| **CRNN-LX** (hepsi) | ✓ | ✓ | ✓ | **80,73** | **9,09** | **+0,38** |

McNemar (eğitim scriptinin CSV bayrakları, N=20.310):

| Karşılaştırma | yalnız A doğru | yalnız B doğru | Δ pp | p |
|---|---:|---:|---:|---:|
| narrow vs full | 784 | 862 | +0,38 | 0,058 |
| narrow vs photo | 822 | 820 | −0,01 | 0,980 |
| narrow vs elastic | 812 | 865 | +0,26 | 0,204 |
| narrow vs morph | 767 | 813 | +0,23 | 0,258 |
| elastic vs full | 798 | 823 | +0,12 | 0,551 |
| morph vs full | 796 | 828 | +0,16 | 0,442 |

Hiçbir çift p<0,01 eşiğini geçmiyor. Beş model 14.474 kelimede (%71,3)
birlikte doğru, 2.483'te (%12,2) birlikte yanlış; 3.353 kelimede en az biri
farklı — modeller **benzer sayıda ama farklı hatalar** yapıyor.

**Sonuç:** Önerilen augmentation tam veride ölçülebilir bir katkı
sağlamıyor (+0,38 pp, anlamsız). Makaledeki +6,48 pp iddiası, yarım veri
(%39 etiket) ve farklı ölçüm hatlarının yapay eseriydi.

## Tablo B — sözlük / trigram ablation (aynı optik model, farklı düzeltme)

Her satır aynı checkpoint'in aynı greedy çıktısına farklı post-processing:

| Düzeltme | narrow | photo | elastic | morph | **full** |
|---|---:|---:|---:|---:|---:|
| yok (ham greedy) | 78,76 | 78,69 | 79,18 | 78,75 | 78,82 |
| + IAM sözlüğü (7.173) | 76,38 | 76,53 | 76,79 | 76,45 | 76,83 |
| + IAM sözlüğü + trigram | 76,64 | 76,76 | 76,93 | 76,81 | 77,04 |
| **+ IAM+NLTK (239.126) + trigram** | **80,35** | **80,34** | **80,68** | **80,64** | **80,73** |

`full` için CER: ham 8,60 → IAM 11,10 → IAM+trigram 10,94 → IAM+NLTK+trigram 9,09.

**Sonuç:** Yalnız eğitim sözlüğüyle düzeltme **zarar veriyor** (−2,0 pp):
geçerli ama eğitimde görülmemiş kelimeler zorla eğitim kelimelerine
çevriliyor. NLTK ile genişletilmiş sözlük ise tutarlı biçimde **+1,9 pp**
kazandırıyor. Düzeltici WA'yı artırırken CER'i hafifçe kötüleştiriyor
(yanlış düzeltmeler yakın-ıska tahminleri tamamen farklı kelimelere çeviriyor).
Test kelimelerinin sözlük kapsaması: yalnız IAM %84,8, IAM+NLTK %94,0.

## Bulgular (makale için)

1. Ana sayı: **CRNN-LX %80,73 WA / %9,09 CER**, N=20.310, Aachen
   writer-disjoint, sözlük yardımlı. Sözlüksüz (ham) **%78,82**.
2. Augmentation katkısı istatistiksel olarak sıfırdan ayırt edilemiyor.
   Birinci katkı iddiası düşüyor; dürüst ifade: "geniş fotometrik, elastik
   ve morfolojik dönüşümler, 48k kelimelik eğitim kümesinde ölçülebilir
   iyileşme sağlamadı".
3. Orijinal koddaki elastik deformasyon ~0,05 px RMS ile fiilen etkisizdi;
   burada düzeltilmiş genlik (1–3 px RMS) kullanıldı ve yine anlamlı fark
   çıkmadı.
4. Ayakta kalan katkı: **sözlük kapsaması analizi** — eğitim sözlüğüyle
   düzeltme zararlı, genel İngilizce sözlükle +1,9 pp; ve bunun CER
   pahasına geldiği.
5. Literatürle karşılaştırma artık aynı test kümesinde (336 form): Kang 2018
   82,55; Kang 2021 84,09; AttentionHTR 84,60 — hepsi bizim 80,73'ün
   üstünde. Makale "denk/üstün" değil "basit CRNN ile 2–4 pp geride,
   sıfır dış veriyle" diye konumlanmalı.
6. Aynı protokolde bizden **kötü** olan tek yayın: Sueiras vd. 2018
   (Neurocomputing, seq2seq+attention). Bölmeleri 47.952/20.306/7.558 =
   Aachen `ok`-filtreli train/test/val (47.999/20.310/7.559); sözlüksüz
   WER 23,8 / CER 8,8 (dört bağımsız kaynakta aynı: Dutta 2018 Tab. III,
   Kang 2021 Tab. 7, Kass&Vats 2022 Tab. 5, Mondal 2022 Tab. 1). Bizim
   sözlüksüz 78,82 → +2,6 pp; sözlüklü 80,73 → +4,5 pp. Makalede
   Discussion §VI.C artık bu karşılaştırma.
7. Eski Tablo IV'teki "Dutta 2018 = 77,14" satırı yanlış atıftı: o sayı
   HWRCNet'in (Rajesh 2022, Tab. 2) kendi 95:5 bölmesinde yeniden eğittiği
   CNN-RNN; Dutta'nın kendi sayısı 12,61 WER (büyük/küçük harf ve
   noktalama yok, sentetik ön-eğitim, TTA). Satır "re-trained in [Rajesh]"
   olarak düzeltildi; Mondal 2022 (YOLOv3, sözlüksüz 70,79) eklendi.

## Düzeltme (12 Eylül) — n-gram katkısı ve tek kaynak

Tablo B'de **eksik bir hücre** vardı: genişletilmiş sözlük *trigram olmadan*
hiç ölçülmemişti, dolayısıyla trigram'ın katkısı yalnızca 7K sözlükte
biliniyordu. Ölçülünce makaledeki "belirleyici olan kapsama, n-gram değil"
iddiasının **yanlış** olduğu çıktı.

`cloud/ablation_lexicon_all.py` ile beş model x beş düzeltme, tek süreçte,
fp32 (deterministik: iki bağımsız koşu 20.310/20.310 aynı):

| Düzeltme | narrow | photo | elastic | morph | **full** |
|---|---:|---:|---:|---:|---:|
| yok (ham greedy) | 78,76 | 78,68 | 79,18 | 78,75 | 78,83 |
| IAM sözlüğü (7.173), sadece edit | 76,14 | 76,26 | 76,46 | 76,25 | 76,42 |
| IAM sözlüğü + n-gram | 76,64 | 76,75 | 76,94 | 76,80 | 77,04 |
| **genişletilmiş (239.126), sadece edit** | 79,27 | 79,23 | 79,50 | 79,37 | **79,51** |
| genişletilmiş + n-gram (CRNN-LX) | 80,36 | 80,33 | 80,67 | 80,64 | **80,74** |

Türetilen katkılar (pp):

| | narrow | photo | elastic | morph | full |
|---|---:|---:|---:|---:|---:|
| IAM sözlüğü vs yok | −2,62 | −2,42 | −2,71 | −2,50 | −2,42 |
| n-gram @ IAM sözlüğü | +0,50 | +0,49 | +0,47 | +0,55 | +0,63 |
| genişletilmiş vs yok | +0,51 | +0,55 | +0,32 | +0,63 | +0,67 |
| **n-gram @ genişletilmiş** | +1,09 | +1,10 | +1,18 | +1,26 | **+1,23** |
| toplam düzeltici | +1,60 | +1,65 | +1,50 | +1,89 | +1,91 |

**Sonuç:** kapsama işaretin yönünü belirliyor, frekans önceliği ise
büyüklüğün çoğunu sağlıyor; ve n-gram'ın değeri sözlük büyüdükçe artıyor
(7K'da ~+0,5, 239K'da ~+1,2), çünkü büyük sözlükte aynı edit mesafesindeki
aday sayısı artıyor ve sıralama kritik hale geliyor.

### Yan bulgular

1. **Berabere adaylarda seçim kuralı tanımsızdı.** Eski "sadece edit mesafesi"
   kodu eşit mesafeli adaylar arasında Python `set` sırasına göre seçiyordu;
   6.143 benzersiz hipotezin 1.239'unda sonuç değişiyor, WA'yı 0,41 pp
   oynatıyordu (76,83 → 76,42). Yeni kod deterministik (kısa aday önce, sonra
   alfabetik) ve `correct_word`'ün yapısıyla örtüşüyor, böylece n-gram katkısını
   doğru izole ediyor.
2. **AMP (fp16) çıkarımı bit düzeyinde tekrarlanabilir değil**: aynı
   checkpoint'in iki koşusu 20.310 kelimenin 2'sinde farklıydı. fp32'de 0 fark.
3. **Makale iki değerlendirme yolunu karıştırıyordu**: Tablo II'nin WA sütunu
   sözlük-ablation'ından, p sütunu ve hata analizi eğitim script'inin
   CSV'sinden geliyordu; ikisi ~9 kelime farklıydı. Artık her şey tek
   kaynaktan: `results/ablation_lexicon5_all.json` + `results/preds_det/`.

Güncellenen sayılar: WA 80,73→80,74; ham 78,82→78,83; baseline 80,35→80,36;
McNemar p 0,058→0,060 (photo 0,901, elastic 0,122, morph 0,165); hatalı kelime
3.903→3.912.

### Sıfır-augmentation çapası (12 Eylül, eklendi)

Beş konfigürasyonun hepsinde konvansiyonel augmentation açıktı, dolayısıyla
"augmentation işe yarıyor mu" sorusu ölçülmemişti. `--aug-mode none` ile
altıncı model eğitildi (aynı ayarlar: 100 epoch, batch 128, lr 7e-4,
patience 15, seed 42; early stopping epoch 53, en iyi val WA %82,57 @ep38).

| Konfigürasyon | WA | CER | ΔWA vs narrow | p | en iyi val WA |
|---|---:|---:|---:|---:|---:|
| **augmentation yok** | **77,87** | 11,00 | **−2,49** | **2×10⁻²⁷** | 82,57 @38 |
| narrow (konvansiyonel) | 80,36 | 9,34 | — | — | 84,61 @87 |
| + geniş fotometrik | 80,33 | 9,34 | −0,03 | 0,901 | 84,34 @76 |
| + elastik | 80,67 | 9,09 | +0,32 | 0,122 | 84,69 @77 |
| + morfolojik | 80,64 | 9,09 | +0,28 | 0,165 | 84,58 @84 |
| CRNN-LX | 80,74 | 9,09 | +0,38 | 0,060 | 84,77 @70 |

Sözlüksüz fark daha da büyük: 78,76 vs **74,33** (−4,43 pp). Yani sözlük,
zayıf optik modelin açığını kısmen kapatıyor.

**Bu bir pozitif kontrol.** Aynı test, aynı veri, aynı seed konvansiyonel
augmentation'ı p=2×10⁻²⁷ ile yakalıyor; bizim üç dönüşümümüzde ise hiçbir şey
bulmuyor (+0,38 pp, p=0,060). Yani null sonuç "deney duyarsız" değil,
**doygunluk**. Hakem "farkı göremediniz çünkü deneyiniz zayıf" derse cevap bu.

Mekanizma: çapa modeli eğitim kaybını 0,018'e kadar indiriyor (augmentation'lı
koşuların beşte biri), val loss'u yükseliyor, 38. epoch'ta doyup 53'te duruyor
— yani ezberliyor. Augmentation'lı koşular 70–87. epoch'a kadar öğrenmeye
devam ediyor.

Sözlük etkileri çapada daha büyük (optik çıktı zayıf olduğu için düzeltecek
daha çok hata var): IAM sözlüğü −1,08, genişletilmiş +1,43, n-gram +2,10,
toplam **+3,54** pp (augmentation'lı beşlide toplam +1,50…+1,91).

## Dosyalar

- `Model_abl_<mod>/` — `best_model_wa.pth`, `test_results_analysis.csv`
  (kelime bazlı), `training_history.json`, `training_log.csv`, `results.json`
- `results/ablation_lexicon5_all.json` — **tek geçerli kaynak**: 6 optik model
  x 5 düzeltme, fp32, deterministik, tek değerlendirme geçişi
- `results/preds_det/preds_<mod>.csv` — kelime bazlı tahminler (McNemar, hata
  analizi ve Şekil 4 bunlardan üretiliyor)
- `results/_superseded/` — eski per-mode JSON'lar; AMP, tanımsız beraberlik
  kuralı ve eksik hücre içeriyorlar, **atıf yapılmamalı** (gerekçe o klasörün
  README'sinde)

## Eski 84,54 sayısı neden yeniden üretilemiyor (7 Eylül, ek ölçüm)

Aynı yerel ortam, orijinal ön işleme (native → 32×128), AMP, her modele kendi
eğitim sözlüğü (+NLTK):

| Model | Sözlük | Eski alt küme (87 form, 5.338) | Tam test (336 form, 20.310) |
|---|---|---:|---:|
| Eski (Berhat'ın ağırlıkları, 31.615 kelimeyle) | yok | 74,97 | 74,55 |
| | +lexicon | **78,34** | 77,32 |
| Yeni CRNN-LX (47.997 kelimeyle) | yok | 79,19 | 78,15 |
| | +lexicon | **81,21** | 80,37 |

- Berhat'ın ağırlıkları yerelde iki bağımsız scriptle 78,29 ve 78,34 veriyor;
  Kaggle'da raporlanan 84,54 hiçbir yerel ölçümde çıkmadı. Kaggle notebook'una
  erişim yok; en olası açıklama sözlüğün Kaggle'daki tam `words.txt`'den
  (test transkripsiyonları dahil) kurulmuş olması, ama kanıtlanamıyor.
- Yeni model her koşulda eskisinden 2,9–3,6 pp iyi (daha fazla eğitim verisi).
- Eski alt küme yalnızca c/d/e kategorilerinden (d: %76); tam test 8 kategoriye
  yayılıyor. Yeni model eski alt kümede 81,2, tam testte 80,4–80,7: alt küme
  ~0,8 pp daha kolay.
- Yeni model legacy ön işlemeyle 80,37, kendi 64×256 yoluyla 80,73: eğitimle
  aynı yol ~0,35 pp daha iyi (beklenen).

Sonuç: 84,54 → 80,73 bir düşüş değildir; ilki yeniden üretilemeyen bir sayı,
ikincisi dört kat büyük ve daha çeşitli bir test kümesinde doğrulanmış sayıdır.

## 15 Eylül revizyonu (hoca)

- Başlık: *Decomposing Lexicon-Assisted Correction for Isolated Handwritten
  Word Recognition on IAM: Effects of Lexicon Coverage and Frequency-Based
  Ranking*. §6.1 → "Interpretation of the Augmentation Results", §6.3 →
  "Comparison with Previous Word-Level HTR Systems".
- Terminoloji: "trigram/n-gram" → **unigram frekans önceliği**. Kodda
  `score_word` hep `prev_words=None` ile çağrılıyor; bigram/trigram dalları
  hiç çalışmıyor. Test kelimelerinin sadece %34'ünün önceki iki kelimesi
  eğitim metninde görülmüş bir trigram bağlamı oluşturuyor.
- Tek-seed ifadeleri yumuşatıldı (özet, katkı 2, §5.1, §6.1, §6.3, sonuç).
- Tablo 2: Δ_B (CRNN-B'ye göre) ve Δ_P (+geniş fotometriğe göre) ayrı
  sütunlar; ikisi de yuvarlanmamış doğruluklardan.
- Sonuç: "2–4 pp" → "1,8–3,9 pp".
- Ek düzeltmeler: morfolojik olasılık 0,15 (0,3 değil); early stopping
  val loss **veya** val WA iyileşince sıfırlanır; eski elastik ayarı 64×256'da
  eksen başına 0,02–0,04 px RMS (max 0,19 px).
- **Bekleyen:** seed tekrarları (CRNN-B ve CRNN-LX × seed 123, 456) —
  Berhat, Kaggle, `cloud/ABLATION_REHBER.md` §10.

## Gerçek trigram + satır bağlamı (15 Eylül, akşam) — yeni önerilen düzeltici

Unigram önceliğinin yerine **interpolated Kneser-Ney trigram** (D=0,75)
kondu (`cloud/kn_trigram.py`). Bağlam: aynı satırdaki önceki iki kelime
kırpıntısı için **sistemin kendi çıktısı** (asla etiket). Satır başında ve
satırda eksik kelime (status≠ok) olan yerde bağlam sıfırlanır. Aday kümesi ve
kabul kuralı unigram düzelticiyle birebir aynı; sadece sıralama değişti.
α (edit cezası) doğrulamada {3,5,7,10} arasından seçildi, testte bir kez.

Doğrulama (`cloud/kn_trigram_selftest.py`): P(w), P(w|h), P(w|h1 h2)
271.303 kelimelik sözlükte **tam 1,000000000000**'e toplanıyor (görülmüş ve
görülmemiş bağlamlarda); görülmemiş bağlam alt mertebeye birebir düşüyor;
çıktı satır işleme sırasından bağımsız ve deterministik.

CRNN-LX optik modeli sabit, 239K sözlük (`results/ablation_trigram.json`):

| Sıralama (derlem) | α | test WA | CER | unigram'a göre | p |
|---|---:|---:|---:|---|---:|
| add-one unigram (IAM) — eski | 5 | 80,74 | 9,09 | ref. | — |
| KN unigram (IAM) | 5 | 80,74 | 9,09 | +19 / −18 kelime | 1,0 |
| KN trigram + bağlam (IAM) | 7 | 80,84 | 9,07 | +45 / −25 | 0,022 |
| KN unigram (IAM+Brown) | 7 | 81,32 | 8,73 | +155 / −36 | 9×10⁻¹⁹ |
| **KN trigram + bağlam (IAM+Brown)** | 7 | **81,66** | **8,58** | **+220 / −32** | **1×10⁻³⁵** |
| kâhin: bağlam = gerçek etiketler (rapor edilmez) | 7 | 81,73 | 8,55 | | |

Ayrıştırma: derlem +0,59, bağlam +0,34 (ters sırada bağlam +0,10, derlem
+0,83 — bağlam ancak derlem büyükken işe yarıyor). Kâhinle fark sadece
0,06 pp: kendi hatalarımızın bağlamı bozması neredeyse hiç kayıp yaratmıyor.
Dört α değerinin hepsi testte 81,43–81,66 veriyor. CER ilk kez sözlüksüz
değerin (8,60) **altına** iniyor.

**Sızıntı kontrolü** (`cloud/brown_leakage_check.py` →
`results/brown_leakage.json`): IAM test satırlarındaki 5-gram'ların %0,63'ü,
6-gram'ların %0,11'i Brown'da birebir geçiyor, 8-gram hiç yok; en uzun ortak
dizi 7 kelime ve hepsi deyim ("in spite of the fact that", "on the other
hand"). Brown test metnini içermiyor.

Dürüstlük notu makalede: bağlam, IAM kelimeleri satırlardan kesildiği için
var; gerçekten tek başına kelimeler için geçerli sayı bağlamsız 80,74.
Tablo 4'te iki satır da var.

### Tablo 2 sol-bağlamlı trigramla (ara adım; geçerli sayılar artık `results/ablation_viterbi.json` içindeki `KN3-left` girdileri)

Altı model, fp32, α her model için doğrulamada seçildi (hepsinde 7).
İki bağımsız geçiş 20.310/20.310 aynı.

| Konfigürasyon | greedy | unigram | **trigram+bağlam** | CER | Δ_B | p (vs CRNN-B) |
|---|---:|---:|---:|---:|---:|---:|
| augmentation yok | 74,33 | 77,87 | **78,81** | 10,53 | −2,41 | 8×10⁻²⁷ |
| CRNN-B | 78,76 | 80,36 | **81,22** | 8,91 | ref. | ref. |
| + geniş fotometrik | 78,68 | 80,33 | **81,29** | 8,84 | +0,06 | 0,760 |
| + elastik | 79,17 | 80,67 | **81,60** | 8,60 | +0,38 | 0,055 |
| + morfolojik | 78,75 | 80,64 | **81,48** | 8,68 | +0,26 | 0,189 |
| CRNN-LX | 78,83 | 80,74 | **81,66** | 8,58 | +0,44 | **0,025** |

**Dikkat:** CRNN-LX − CRNN-B farkı trigramla p=0,025'e indi (unigramla
0,060). Makalenin eşiği p<0,01 ve beş karşılaştırma için Bonferroni de
0,01 → hâlâ anlamlı değil, ama p<0,05'i geçiyor. Makalede açıkça yazıldı,
"çözülmemiş" diye sunuldu. Seed tekrarları bunu belirleyecek.

Kelime bazlı (`cloud/paper_stats_trigram.py` → `results/paper_stats_trigram.json`):
CRNN-LX 3.724 hata; düzeltici 2.938 hipotezi değiştiriyor, 1.076 düzeltiyor,
501 bozuyor. Unigram aynı 2.938'i değiştirip aynı 501'i bozuyor ama sadece
888 düzeltiyor — fark tamamen doğru adayı seçmekten geliyor.

## Satırın tamamı (sağ + sol bağlam, Viterbi) — 16 Eylül, nihai düzeltici

`cloud/kn_trigram.py` → `LineViterbiCorrector`: satırdaki her kelimenin aday
sütunu üzerinden Σ[log P_KN(w_i | w_{i-2}, w_{i-1}) − α·d_i]'yi **tam** en
büyükleyen dizi (ikinci dereceden Viterbi). w_i, w_{i+1} ve w_{i+2}'nin
terimlerine de girdiği için her seçim hem soldaki hem sağdaki komşuya bağlı.
Doğrulama: 400 rastgele satır düzeninde kaba kuvvet aramayla birebir aynı
(`cloud/viterbi_selftest.py`); sıra bağımsız; deterministik.

Seçenekler (doğrulamada seçildi, α ızgarası 1–30; `cloud/ablation_viterbi.py`
→ `results/ablation_viterbi.json`, CRNN-LX):

| Çözücü | α | val | test WA | CER | sola göre |
|---|---:|---:|---:|---:|---|
| soldan sağa (önceki) | 7 | 85,50 | 81,66 | 8,58 | ref. |
| tüm satır | 10 | 85,66 | 81,77 | 8,54 | +56/−34, p=0,026 |
| **tüm satır + keep-OOV** (seçilen) | 2 | **85,98** | **81,95** | 8,54 | +394/−335, p=0,032 |
| tüm satır + real-word | 10 | 85,69 | 81,94 | 8,51 | +141/−86, p=0,0003 (seçilmedi) |
| ikisi birlikte | 10 | 84,50 | 79,71 | 8,42 | çöküyor |

keep-OOV: sözlük dışı hipotez d=0 aday olarak kendini koruyabilir (taban
olasılık). real-word: sözlükteki kelime de 1 mesafedeki komşularıyla yarışır.
Seçilen çözücü altı modelde: çapa +0,08 (p=0,57); CRNN-B +0,42 (0,002);
photo +0,44 (0,003); elastic +0,33 (0,025); morph +0,32 (0,034);
CRNN-LX +0,29 (0,032). Kazanç muhafazakârlıktan: 1.925 değişiklik, 815
düzeltme / 181 bozma (soldan sağa: 2.939 / 1.076 / 501).

**Tablo 2 (nihai düzeltici):** CRNN-LX − CRNN-B = +0,32, p=0,132
(unigram 0,38/0,060; sol trigram 0,44/0,025). Üç düzelticide de aynı işaret,
hiçbirinde p<0,01 yok. Çapa −2,74, p=1×10⁻³⁰. Tablo 4: Kang18'in 0,6,
Kang21'in 2,1, AttentionHTR'nin 2,7 pp altında; Sueiras'ın 5,8 üstünde.

**Süreçler arası determinizm:** fp32 tek süreçte 20.310/20.310 aynı, ama iki
ayrı süreç arasında 1 kelime ('phywied'/'phycied', ikisi de yanlış) değişti —
cuDNN algoritma seçimi. `hypotheses()`'e `cudnn.deterministic=True` eklendi
(`cloud/determinism_probe.py` ile iki ayrı süreçte aynı sha256 doğrulandı) ve
üç puanlama script'i bu ayarla yeniden koşuldu.


## Seed tekrarları (20 Eylül) — hocanın 5. maddesi kapandı

Berhat, CRNN-B ve CRNN-LX'i **seed 123 ve 456** ile Kaggle T4'te yeniden
eğitti (`brht25/seed1-output`, `brht25/seed2-output` — public). Dört ağırlık
uzaktan zip okumasıyla indirildi (4,1 GB paketten 816 MB;
`scratchpad/kaggle_partial.py` mantığı), `Model_seed_{narrow,full}_{123,456}/`
altına konuldu ve makaledeki **aynı** deterministik yolla, **aynı nihai
düzelticiyle** (239K sözlük + KN trigram IAM+Brown + tüm-satır Viterbi,
keep-OOV) puanlandı: `cloud/ablation_viterbi.py --modes ... --baseline narrow`
→ `results/ablation_viterbi_seeds.json`, `results/preds_viterbi_seeds/`.

Doğrulamalar: (a) Berhat'ın `aachen_splits/*` dosyaları bizimkiyle **birebir
aynı** (yalnız satır sonu CRLF/LF farkı); (b) dört koşunun doğrulama eğrileri
birbirinden farklı, yani `--seed` gerçekten verilmiş; (c) seed 42 modelleri bu
koşuda da aynı sayıları verdi (CRNN-B 81,64 / CRNN-LX 81,95), yani puanlama
yolu değişmedi; (d) ikinci çözüm geçişi 20.310/20.310 aynı.

| Seed | CRNN-B WA | CRNN-LX WA | Δ (LX−B) | McNemar p | CER B / LX |
|---|---:|---:|---:|---:|---|
| 42 (yerel, RTX 4070) | 81,64 | 81,95 | +0,32 | 0,132 | 8,83 / 8,54 |
| 123 (Kaggle T4) | **81,93** | 81,43 | **−0,49** | **0,024** | 8,20 / 8,91 |
| 456 (Kaggle T4) | 81,66 | **82,44** | **+0,78** | **3,9×10⁻⁴** | 8,53 / 8,28 |
| **ortalama ± SD** | **81,74 ± 0,16** | **81,94 ± 0,50** | **+0,20 ± 0,64** | 0,64 (eşleştirmeli t, t(2)=0,54) | 8,52 ± 0,32 / 8,57 ± 0,32 |

Sözlüksüz (greedy): CRNN-B 78,82 ± 0,19; CRNN-LX 78,92 ± 0,62.
Eğitim özetleri: narrow 84,61@87 / 84,69@97 / 85,08@77; full 84,77@70 /
84,23@53 / 84,84@84 (en iyi val WA @epoch).

**Sonuç — augmentation farkı çözülmedi, hem de net biçimde:**

1. Fark **işaret değiştiriyor**: +0,32 / −0,49 / +0,78 pp.
2. Üç seedin ikisinde p<0,05 çıkıyor ama **zıt yönlerde** (123'te CRNN-B
   anlamlı biçimde iyi, 456'da CRNN-LX). Yani tek seedle yapılan McNemar
   testi 20 bin kelimede her iki yönde de "anlamlı" sonuç üretebiliyor.
3. Aynı konfigürasyonun seedler arası yayılımı **1,00 pp** (CRNN-LX
   81,43–82,44) — ölçmeye çalıştığımız etkinin birkaç katı.
4. Konvansiyonel augmentation'ın etkisi (−2,74 pp, p=1×10⁻³⁰) bu gürültü
   tabanının bir mertebe üstünde, yani pozitif kontrol etkilenmiyor.

Makaleye giren: yeni **Tablo 3** (seed tekrarları), §5.1'de üç seed paragrafı,
özet/katkı 2/§6.1/sonuç/tehditler güncellendi; §4'e "seed 123 ve 456 farklı
makinede (T4), aynı kod sürümü ve aynı hiperparametrelerle" notu eklendi.

**Bekleyen tek doğrulama:** Berhat'ın komut satırında
`--elastic-legacy-amplitude 0 --elastic-alpha 1 3` var mıydı? Veri setinde
çıktı günlüğü yok, dosyalardan doğrulanamıyor. Kayıp değerleri
(full: 0,043/0,020 vs narrow: 0,013/0,014) augmentation'ın açık olduğunu
gösteriyor ama elastik genliğin düzeltilmiş olup olmadığını ayırt etmiyor.
Berhat onaylayınca bu not silinecek; onaylamazsa `full` seed satırları
"elastik ayarı doğrulanmadı" diye işaretlenmeli.
