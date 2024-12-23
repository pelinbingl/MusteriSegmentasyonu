# KMeans ile Müşteri Segmentasyonu

Bu repository, **KMeans algoritması** kullanılarak yapılan bir **müşteri segmentasyonu** projesini içerir. Projenin temel amacı, müşterilerin demografik ve finansal verilerine dayalı olarak gruplandırılması ve şirketlerin farklı müşteri grupları için stratejiler geliştirmesine olanak sağlamaktır.

---

## Özellikler

- **Veri Ön işleme**:
  - Kategorik verilerin (Cinsiyet gibi) sayısal verilere dönüştürülmesi.
  - Verilerin KMeans algoritması için ölçeklendirilmesi.
- **Kümeleme**:
  - KMeans algoritması kullanılarak müşterilerin segmentlere ayrılması.
  - **PCA (Principal Component Analysis)** yardımıyla küme görseli oluşturulması.
- **Değerlendirme**:
  - **Silhouette skorları** kullanılarak kümeleme kalitesinin değerlendirilmesi.
- **Etkileşimli Tahmin**:
  - Kullanıcıdan yeni veri alıp, müşteri segmentini tahmin etme ve sonucu görselleştirme.

---

## Gerekli Kütüphaneler

Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Veri Seti

Bu projede kullanılan veri setinde şu özellikler yer almaktadır:
- **Gender** (kategorik): Erkek ya da Kadın.
- **Age** (sayısal): Müşterinin yaşı.
- **Annual Income (k$)** (sayısal): Müşterinin yıllık geliri (bin dolar cinsinden).

Veri setinin doğru dosya yolunda olduğundan emin olun:

```python
url = "C:\\Segmentasyon\\Mall_Customers.csv"
```

---

## Kodun Genel Yapısı

### 1. Veri Yükleme ve Ön işleme
- **Kategorik Verilerin Dönüştürülmesi**:
  Cinsiyet, `LabelEncoder` kullanılarak sayısal verilere dönüştürülür. ("Kadın = 0", "Erkek = 1")
- **Özellik Ölçeklendirme**:
  `StandardScaler` kullanılarak veriler ölçeklendirilir, bu da kümeleme performansını arttırır.

### 2. KMeans Kümeleme
- Veriler, varsayılan olarak **4 küme** üretecek şekilde gruplandırılır.
- Eğitim ve test setleri için **Silhouette skorları** hesaplanarak model performansı değerlendirilir.

### 3. Görselleştirme
- PCA kullanılarak veri 2 boyuta indirgenir.
- Küme merkezleri, test verileri, eğitim verileri ve kullanıcıdan alınan yeni veriler bir arada gösterilir.

### 4. Yeni Veriyi Tahmin Etme
- Kullanıcıdan yaş, yıllık gelir ve cinsiyet bilgisi alınır.
- Bu veriler için küme tahmini yapılır.
- Yeni veri mevcut küme yapısı içerisinde görselleştirilir.

---

## Temel Sonuçlar

- **Silhouette Skorları**:
  - Eğitim Silhouette Skoru: `0.35`
  - Test Silhouette Skoru: `0.27`
    
    ![Silhouette Skor Grafiği](https://github.com/user-attachments/assets/4b599b5d-e434-4515-a6fb-b89a17a7d78f)
- **Küme Görselleştirmeleri**:
  - Eğitim ve test verileri ile küme merkezlerini gösteren grafikler.
  -![Kümeleme Görselleştirmesi]![image](https://github.com/user-attachments/assets/d6eb2e70-ebd6-481d-b826-f906d8b96e9a)


## Örnek

### Kullanıcı Girdisi:
```
Değerleri girin: 25, 50, 1
```

### Çıktı:

![Tahmin Görseli]![image](https://github.com/user-attachments/assets/5a21e56b-3f06-4ab8-9744-110910687c40)

Bu müşteri şu kümeye ait: `0`

---

