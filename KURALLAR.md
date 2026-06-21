# Proje Kuralları ve Yöntem Dikkat Edilecekler

Bu dokümana yeni bir şey denemeden önce **mutlaka bak**. Yapılan tüm
deneyler, makaledeki tüm sayılar bu kurallara uymalı — aksi takdirde
reviewer veya hoca reject eder.

---

## 1. NE İÇİN OPTİMİZE EDİYORUZ?

### Asıl Metrik: Aachen Test Set Word Accuracy
- **5,338 sample** üzerinde
- **Writer-disjoint** — test yazarları train'de YOK
- Şu anki en iyi: **76.85%** (V2 + Beam + V3 Extended Trigram)
- **Hedef: %85+** (pure CRNN için zor ama mümkün)

### Yardımcı Metrikler (Karar Verirken Bakılır)
- **Wilson 95% CI** — single number değil, aralık raporla
- **McNemar p-value** — değişiklik istatistiksel anlamlı mı?
- **Mean Character Accuracy** — WA katı bir metric, char acc model'in
  gerçek öğrenmesini gösterir
- **CER (Character Error Rate)** — düşük olmalı

### Optimize ETMEDIĞIMIZ
- ❌ **Validation WA**: checkpoint seçimi için kullanılır, ama final
  rapor için **test set** lazım. Val WA test'ten ~3-5pp yüksek olur,
  bu doğal (early stopping val'de yapılıyor)
- ❌ **Train accuracy**: %100'e gidebilir (overfitting), önemsiz

---

## 2. YASAK ŞEYLER (Kesin Yapmamamız Gereken)

### 2.1 Test Set Leakage
**Yapma**:
- Test set yazarlarının (336 form) HERHANGİ bir kelimesini training'e ekleme
- Test set transcription'larını trigram vocabulary'ye ekleme
- Test set'i hyperparameter tuning için kullanma

**Yapılması Gereken**:
- Test set'e sadece **ONE-SHOT** olarak bak (best checkpoint seçildikten
  sonra, bir kez)
- Test sonucu kötü olursa BURADA dur, başka değişikliği val ile yap

### 2.2 Vocabulary Leak
**Yapma**:
- Test set transcription'larını trigram'ın vocabulary'sine ekleme
  (Custom split'te olan tam buydu, +11pp şişti)
- NLTK + IAM kullanıyoruz — bu OK çünkü test transcription'ları YOK,
  sadece İngilizce dictionary

**Yapılması Gereken**:
- Trigram vocabulary kaynağını **açıkça belirt** (IAM train + NLTK)
- Test set'in word'lerini sözlüğe **asla** ekleme

### 2.3 Cherry-Pick
**Yapma**:
- 10 seed eğit, en iyi seed'i raporla → SELECTION BIAS
- Test set'ten "zor" sample'ları çıkar → SAMPLE BIAS
- En iyi yazarı test'e koy, kötülerini train → DATA SELECTION

**Yapılması Gereken**:
- Ensemble yapacaksak **3-5 seed'in ORTALAMASI** veya **VOTING**
- Tüm sayıları raporla, single best değil

### 2.4 Çoklu Hipotez Testi Olmadan Anlamlılık İddiası
**Yapma**:
- 20 farklı konfigürasyon dene, p<0.05 olanları "anlamlı" diye yaz
  (multiple comparison problem, Bonferroni gerekir)

**Yapılması Gereken**:
- HER ABLATION için **paired McNemar** test
- HEDEFLİ değişiklikleri test et, "ne tutarsa" yöntemini değil

---

## 3. METODOLOJİK KURALLAR

### 3.1 One-Change-At-A-Time
**Doğru**:
1. V2 baseline → 76.85%
2. SADECE trigram'ı V3'e değiştir → 76.85% mi 76.53% mi gör
3. SADECE model'i V3'e değiştir → kazançı izole et
4. Sonra ikisini birleştir

**Yanlış**:
- "Hem model'i büyüttüm hem trigram'ı değiştirdim hem augmentation'ı
  değiştirdim, sonuç +3pp" → hangi katkı hangisinden bilinmez

### 3.2 Baseline'a Karşı Paired Test
Her yeni şey için:
1. Aynı test set'te eski ve yeni'yi koş
2. **Paired McNemar** test (sample-by-sample karşılaştır)
3. Eğer p<0.01 ve Δ>0.5pp → KEEP
4. Eğer p>0.05 → muhtemelen şans, REVERT veya tekrar test et

### 3.3 Reproducibility
**Her deney için kaydet**:
- Random seed (sabit veya 3-5 seed)
- Tüm hyperparameter'lar
- Training script'in git commit hash'i
- Eğitim süresi
- Final val + test sayıları

**Saklama yeri**: `Model_aachen_vX/training_history.json` + git commit message

### 3.4 Honest Reporting
**Doğru**:
- "V2 model + V3 trigram +4.80pp (p=10⁻³³)"
- "V1 trigram Aachen'de -0.36pp (anlamsız fark)"
- "WBS marjinal etki, beklenenden düşük"

**Yanlış**:
- En iyi sayıyı seç, kötülerini sakla
- "Beam search +5pp" demek (gerçek +0.19pp iken)

---

## 4. YENİ BİR ŞEY DENERKEN — Checklist

Yeni bir technique denerken **mutlaka** şu sırayla:

### A. Baseline Belirle
- [ ] Şu anki en iyi konfigürasyon nedir? (V2 + Beam + V3 Trigram = 76.85%)
- [ ] Hangi sample'larda hatalı?

### B. Hipotez Yaz
- [ ] Bu değişiklik hangi error pattern'i çözer?
- [ ] Beklenen kazanç ne kadar? (gerçekçi tahmin)

### C. Sadece Bir Şeyi Değiştir
- [ ] Yeni script ayrı bir dosyada (greedy_aachen_v3.py gibi)
- [ ] Eski script'e dokunma (rollback için)
- [ ] Sadece ONE change

### D. Aynı Protokolde Eval Et
- [ ] Aynı test set (5,338 Aachen sample)
- [ ] Aynı CSV format
- [ ] Aynı McNemar testleri

### E. İstatistiksel Test
- [ ] McNemar paired test
- [ ] Wilson 95% CI
- [ ] Effect size (Δ pp)

### F. Karar
- [ ] Δ > 0.5pp VE p < 0.01 → KEEP, devam et
- [ ] Δ < 0.5pp VEYA p > 0.05 → muhtemelen şans, dikkat
- [ ] Δ < 0 → reject

### G. Commit + Document
- [ ] Git branch ayrı (feature/X-experiment)
- [ ] Commit message: ne yapıldı + sonuç
- [ ] Eğer KEEP edildi: main branch'e merge

---

## 5. SIK YAPILAN HATALAR

### Hata 1: "Daha Karmaşık = Daha İyi" Sanmak
- TrOCR-Large 334M params, %92 WA. Bizim 28M model %85 hedefliyor.
- **Karmaşıklık tek başına accuracy getirmez** — doğru kullanım getirir
- Önce küçük modeli MAX'a çıkar, sonra büyüt

### Hata 2: Val WA'yi Test WA Sanmak
- Val'de 76.69% almak Test'te 71.73% demek (model V2 örneği)
- Gap normaldir, **test sayısı raporlanır**

### Hata 3: P-value Tek Başına Önemli Sanmak
- p < 0.05 + Δ = 0.1pp = bilgilendirici ama önemsiz
- p > 0.05 + Δ = 2pp = teknik anlamsız ama büyük örneklem testin yeniden
- **Hem effect size hem p-value bakılır**

### Hata 4: Hata Analizi Yapmadan Çözüm Bulmaya Çalışmak
- "Accuracy düşük, daha büyük model alalım" → SOLUTION-FIRST
- Doğru: "Hangi sample'larda hatalı? Neden? O sebebi çöz" → PROBLEM-FIRST
- Örneğin: trigram hurt analizi yaparak NLTK extension'ın gerekli
  olduğunu anladık. Daha büyük model bu sorunu çözmezdi.

### Hata 5: GPU Yetmediği İçin Workaround
- Batch size'ı düşürerek "fit" et → gradient noise artar
- Doğrusu: gradient accumulation kullan, effective batch korunur

### Hata 6: Augmentation'ı Çok Agresif Yapma
- Model rotation ±15° görürse cursive ile karıştırır
- ±5° → ±7° geçişimiz V3'te makul
- ±10°+ "data destruction" haline gelir

---

## 6. RAPORLAMA STANDARDI

Her ablation için makaledeki tablo formatı:

```
| Method                  | WA (%) | 95% CI       | Δ vs Baseline | McNemar p |
|------------------------|--------|--------------|---------------|-----------|
| Raw Greedy (baseline)   | 71.73  | [70.51,72.92]| —             | —         |
| + V3 Extended Trigram   | 76.53  | [75.37,77.64]| +4.80pp ✓     | 7.04e-33  |
| + Beam k=10             | 76.56  | [75.40,77.69]| +0.03pp       | 0.89      |
```

### Asla Atılmaması Gerekenler
- Wilson CI
- Δ vs baseline
- McNemar p (paired)
- Effect direction (+/-)

---

## 7. ŞU AN PROJENİN DURUMU

### Çözülen
- ✅ Aachen writer-disjoint split kullanıldı
- ✅ V3 trigram (NLTK + IAM) Aachen'de çalışıyor (+4.80pp)
- ✅ Beam search test edildi, marjinal etki bulundu
- ✅ WBS implement edildi, post-hoc test yapıldı
- ✅ Tüm McNemar testleri scripted

### Devam Eden
- ⏳ V3 model (28.73M) eğitiliyor — hedef %78-82

### Bekleyen (Eğer V3 Yetersizse)
- 🔲 3-seed ensemble (~9 saat)
- 🔲 Synthetic data pretraining (~15 saat)
- 🔲 Test-time augmentation
- 🔲 Pretrained CNN backbone (ResNet18)

---

## 8. HOCA SORARSA NE DERIZ

### "Niye Aachen split?"
→ Writer-disjoint, literatür standardı, custom split'te vocabulary leak vardı.

### "Niye trigram?"
→ §3.5.1'de 4 argüman: edit-distance optimal CTC error mode için,
edge deployment'a uygun (6 MB, no GPU), LLM hallucination riski,
auditability.

### "Niye NLTK eklendi?"
→ V2 trigram (IAM-only) Aachen'de hurt veriyordu — 272 valid English
word OOV sayılıp yanlış kelimelere düzeltiliyordu. NLTK extension
+4.80pp, McNemar p=10⁻³³.

### "Pre-training niye yok?"
→ Resource-efficient claim — bizim katkımız "no pretraining single GPU
under 24h" angle'ı. TrOCR ile yarışmıyoruz, edge HTR baseline'ı sunuyoruz.

### "%85 yerine niye %76?"
→ Honest writer-disjoint evaluation. Custom split'te %89.68 idi ama
vocab leak vardı. Reviewer transparenz değerlendirir.

---

## 9. SONUÇ

**Tek cümleyle**: Hızlı sayılar yerine **dürüst sayılar**, single best
yerine **paired karşılaştırma**, ad-hoc fix yerine **sistemik improvement**.

Bir şeyi denerken **bu dosyayı yeniden oku**. Kuralları ihlal eden bir
şey istemiyorsak, o teknik kullanılmaz.

Sorularını yaz, tartışırız.
