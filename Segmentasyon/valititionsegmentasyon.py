import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Veri yükleme ve hazırlama
url = "C:\\Segmentasyon\\Mall_Customers.csv"  # Dosya yolunu doğru girin
df = pd.read_csv(url)

# Kategorik değişkenleri sayısal verilere dönüştür
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # 'Female' = 0, 'Male' = 1

# Özellikleri seç ve ölçeklendir
X = df[['Age', 'Annual Income (k$)', 'Gender']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test olarak ayır
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# KMeans modeli oluştur ve eğit
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_train)

# Eğitim ve test kümeleri için tahminler
y_train_kmeans = kmeans.predict(X_train)
y_test_kmeans = kmeans.predict(X_test)

# Silhouette Skorları
train_silhouette = silhouette_score(X_train, y_train_kmeans)
test_silhouette = silhouette_score(X_test, y_test_kmeans)

print(f"Eğitim Silhouette Skoru: {train_silhouette:.2f}")
print(f"Test Silhouette Skoru: {test_silhouette:.2f}")

# Görselleştirme: Eğitim, test, küme merkezleri ve kullanıcı girdisi
def visualize_clusters(X_train, X_test, y_train_kmeans, y_test_kmeans, kmeans, new_data=None):
    # PCA ile 2D'ye indirgeme
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(12, 8))

    # Eğitim verisi
    scatter_train = plt.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1], 
        c=y_train_kmeans, cmap='viridis', alpha=0.7, s=100, label="Eğitim Verisi"
    )
    # Test verisi
    scatter_test = plt.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1], 
        c=y_test_kmeans, cmap='cool', alpha=0.7, s=80, label="Test Verisi"
    )
    # Küme merkezleri
    plt.scatter(
        cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
        color='red', marker='X', s=300, label="Küme Merkezleri"
    )

    # Kullanıcı girişi (isteğe bağlı)
    if new_data is not None:
        new_data_pca = pca.transform(new_data)
        plt.scatter(
            new_data_pca[:, 0], new_data_pca[:, 1], 
            color='black', marker='P', s=200, label="Yeni Veri"
        )

    # Renkler ve kümeler için ayrı bir gösterge ekle
    handles, _ = scatter_train.legend_elements(prop="colors")
    cluster_labels = [f"Küme {i}" for i in range(kmeans.n_clusters)]
    legend_colors = plt.legend(handles, cluster_labels, title="Kümeler", loc="upper right", fontsize="small")
    plt.gca().add_artist(legend_colors)

    plt.title("KMeans Kümeleme Sonuçları")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Küme ID")
    plt.legend()
    plt.show()

# Yeni veri tahmin ve görselleştirme fonksiyonu
def predict_new_data():
    print("\nYeni bir veri girerek müşteri segmenti tahmini alabilirsiniz.")
    print("Lütfen aşağıdaki sıralamada değerleri virgül ile ayırarak giriniz: ")
    print("Yaş, Yıllık Gelir (k$), Cinsiyet (0: Female, 1: Male)")

    user_input = input("Değerleri girin: ")
    try:
        new_data = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)
        prediction = kmeans.predict(new_data_scaled)
        print(f"Bu müşteri şu kümeye ait: {prediction[0]}")

        # Yeni veriyi görselleştir
        visualize_clusters(X_train, X_test, y_train_kmeans, y_test_kmeans, kmeans, new_data=new_data_scaled)
    except ValueError:
        print("Hatalı giriş yaptınız. Lütfen sayısal değerler girin ve virgül ile ayırın.")

# Sonuçları karşılaştırmak için Silhouette Skoru görselleştirme
def plot_silhouette_scores(train_score, test_score):
    plt.figure(figsize=(6, 4))
    scores = [train_score, test_score]
    labels = ['Eğitim', 'Test']
    plt.bar(labels, scores, color=['blue', 'green'])
    plt.title("Silhouette Skorları")
    plt.ylabel("Silhouette Skoru")
    plt.ylim(0, 1)
    plt.show()

# Görselleştirme ve tahmin
plot_silhouette_scores(train_silhouette, test_silhouette)
visualize_clusters(X_train, X_test, y_train_kmeans, y_test_kmeans, kmeans)
predict_new_data()
