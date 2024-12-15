import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score

# Veri yükleme ve hazırlama
url = "C:\\Segmentasyon\\Mall_Customers.csv" # Dosya yolunu doğru girin
df = pd.read_csv(url)

# Kategorik değişkenleri sayısal verilere dönüştür
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # 'Female' = 0, 'Male' = 1

# Özellikleri seç ve ölçeklendir
X = df[['Age', 'Annual Income (k$)', 'Gender']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross-Validation için ayarlar
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 katlama
fold_no = 1
cv_accuracies = []
cv_losses = []

for train_index, val_index in kf.split(X_scaled):
    print(f"Fold {fold_no} başlıyor...")

    # Train ve Validation Set'leri oluştur
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]

    # KMeans modeli oluştur ve eğit
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_train)

    # Eğitim ve doğrulama kümeleri için tahminler
    y_train_kmeans = kmeans.predict(X_train)
    y_val_kmeans = kmeans.predict(X_val)

    # Silhouette Skoru (Eğitim ve Doğrulama için)
    train_silhouette = silhouette_score(X_train, y_train_kmeans)
    val_silhouette = silhouette_score(X_val, y_val_kmeans)
    
    print(f"Fold {fold_no} - Eğitim Silhouette Skoru: {train_silhouette:.2f}")
    print(f"Fold {fold_no} - Doğrulama Silhouette Skoru: {val_silhouette:.2f}")
    
    # Kayıtlar
    cv_accuracies.append(train_silhouette)
    cv_losses.append(val_silhouette)

    # Bir sonraki fold için hazırlan
    fold_no += 1

# 4. CV Sonuçları
print("\nCross-Validation Sonuçları:")
print(f"Ortalama Eğitim Silhouette Skoru: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Ortalama Doğrulama Silhouette Skoru: {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")

# 5. Sonuçları görselleştir
plt.figure(figsize=(8, 5))
plt.plot(cv_accuracies, label="Eğitim Silhouette Skoru")
plt.plot(cv_losses, label="Doğrulama Silhouette Skoru")
plt.title("Cross-Validation Sonuçları")
plt.xlabel("Fold")
plt.ylabel("Silhouette Skoru")
plt.legend()
plt.show()