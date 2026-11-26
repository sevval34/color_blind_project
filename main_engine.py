import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import time

def apply_kmeans_reduction(image_path, k=8):
    print(f"Resim yükleniyor... Hedef renk sayısı (k): {k}")
    
    # 1. Resmi açma işlemi
    original_img = Image.open(image_path)
    original_img = original_img.convert('RGB')
    
    # Hızlandırmak için resmi biraz küçültme işlemi
    # Eğitim aşaması çok uzun sürmesin diye genişliği 400px'e oranladım
    w, h = original_img.size
    resize_ratio = 400 / w
    small_w, small_h = int(w * resize_ratio), int(h * resize_ratio)
    img_small = original_img.resize((small_w, small_h))
    
    # 2. Veriyi Hazırla (Matrix Flattening)
    # Resmi (Yükseklik, Genişlik, 3) formatından -> (Piksel Sayısı, 3) listesine çevirme işlemi
    # Çünkü AI algoritmaları tablo sever.
    image_array = np.array(img_small)
    pixel_list = image_array.reshape(-1, 3)

    print("Yapay zeka (K-Means) renkleri öğreniyor... (Bu işlem 5-10 saniye sürebilir)")
    start_time = time.time()

    # 3. K-Means Modelini Kur ve Eğit (AI KISMI BURASI)
    # n_clusters=k : Kaç renge indirmek istiyoruz?
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixel_list) # Renkleri öğren!

    end_time = time.time()
    print(f"Öğrenme tamamlandı! Süre: {end_time - start_time:.2f} saniye.")

    # 4. Görüntüyü Yeniden Oluştur
    # Her pikselin yerine, ona en yakın 'Merkez Rengi' koyuyoruz.
    new_colors = kmeans.cluster_centers_.astype('uint8') # Öğrenilen 8 renk
    labels = kmeans.labels_ # Her pikselin hangi renge atandığı
    
    # Yeni pikselleri oluştur
    quantized_pixels = new_colors[labels]
    
    # Listeyi tekrar resim şekline (H, W, 3) sok
    quantized_image_array = quantized_pixels.reshape((small_h, small_w, 3))
    
    # 5. Sonucu Göster
    result_img = Image.fromarray(quantized_image_array)
    
    # Orijinal boyutuna geri büyüt (Görmek kolay olsun diye)
    result_img = result_img.resize((w, h), Image.NEAREST)
    
    result_img.show()
    result_img.save(f"kmeans_result_k{k}.jpg")
    print("✅ İşlem bitti! Sadeleştirilmiş resim kaydedildi.")

# Çalıştır
if __name__ == "__main__":
    # k=8 ile dene (Resmi sadece 8 renkle çizecek)
    apply_kmeans_reduction("images/elma.png", k=8)