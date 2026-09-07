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
| narrow (CRNN-L) | 100 | 84,61 | 87 | 84,58 | epoch sınırı |
| photo | 91 | 84,34 | 76 | 83,75 | 76+15 |
| elastic | 92 | 84,69 | 77 | 84,40 | 77+15 |
| morph | 99 | 84,58 | 84 | 84,40 | 84+15 |
| full (AugCRNN-T) | 85 | 84,77 | 70 | 84,41 | 70+15 |

## Tablo A — augmentation ablation (test, N=20.310, greedy + IAM+NLTK trigram)

Nihai sayılar `results/ablation_lexicon_<mod>.json` (deterministik beraberlik
çözümü, `075688d`). Wilson %95 aralığı yaklaşık ±0,55 pp.

| Konfigürasyon | wide photo | elastic | morph | WA (%) | CER (%) | Δ vs CRNN-L |
|---|:---:|:---:|:---:|---:|---:|---:|
| CRNN-L (baseline) | ✗ | ✗ | ✗ | 80,35 | 9,34 | — |
| + wide photometric | ✓ | ✗ | ✗ | 80,34 | 9,34 | −0,01 |
| + elastic | ✓ | ✓ | ✗ | 80,68 | 9,09 | +0,33 |
| + morphological | ✓ | ✗ | ✓ | 80,64 | 9,09 | +0,29 |
| **AugCRNN-T** (hepsi) | ✓ | ✓ | ✓ | **80,73** | **9,09** | **+0,38** |

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

1. Ana sayı: **AugCRNN-T %80,73 WA / %9,09 CER**, N=20.310, Aachen
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

## Dosyalar

- `Model_abl_<mod>/` — `best_model_wa.pth`, `test_results_analysis.csv`
  (kelime bazlı), `training_history.json`, `training_log.csv`, `results.json`
- `results/ablation_lexicon_<mod>.json` — Tablo B, nihai
- `results/ablation_lexicon.json` — full için ilk (deterministik olmayan) ölçüm
