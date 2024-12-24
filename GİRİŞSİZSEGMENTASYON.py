import pandas as pd
import numpy as np  # numpy kütüphanesini içeri aktar
import matplotlib.pyplot as plt  # matplotlib ile grafikler ekleyebiliriz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # Kümeleme için KMeans algoritması
from sklearn.decomposition import PCA  # Kümeleme için 2D görselleştirme
from sklearn.preprocessing import LabelEncoder

# Dosyadan veri yükleme (CSV formatında olmalı)
url = "C:\\Segmentasyon\\Mall_Customers.csv"
df = pd.read_csv(url)

# Kategorik verileri sayısal verilere dönüştür (Label Encoding)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # 'Female' = 0, 'Male' = 1

# Özellikler (X) ve hedef değişken (y)
X = df[['Age', 'Annual Income (k$)', 'Gender']].values  # 'Age', 'Annual Income (k$)', 'Gender'

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans kümeleme modeli oluştur
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)  # Kümeleme işlemi
y_kmeans = kmeans.predict(X_scaled)  # Küme tahminleri

# Kullanıcıdan yeni veri alma fonksiyonu
def predict_new_data():
    print("\nYeni bir veri girerek müşteri segmenti tahmini alabilirsiniz.")
    print("Lütfen aşağıdaki sıralamada değerleri virgül ile ayırarak giriniz: ")
    print("Yaş, Yıllık Gelir (k$), Cinsiyet (0: Female, 1: Male)")

    user_input = input("Değerleri girin: ")
    try:
        # Kullanıcıdan alınan veriyi işleme
        new_data = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)  # Veriyi ölçeklendir
        prediction = kmeans.predict(new_data_scaled)  # Küme tahminini yap
        print(f"Bu müşteri şu kümeye ait: {prediction[0]}")
    except ValueError:
        print("Hatalı giriş yaptınız. Lütfen sayısal değerler girin ve virgül ile ayırın.")

# Kullanıcı tahmin fonksiyonunu çağır
predict_new_data()

# Eğitim verisinin kümeleme ile görselleştirilmesi
def plot_kmeans_clusters(X, y_kmeans):
    pca = PCA(n_components=2)  # Veriyi 2D'ye indirger
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', s=100)
    plt.title("KMeans Kümeleme Sonuçları")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Küme ID")
    plt.show()

plot_kmeans_clusters(X_scaled, y_kmeans)  # Tüm veriyi kümelerle görselleştir
