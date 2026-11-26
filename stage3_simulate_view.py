#bu dosyadaki kodlarım bilgisayara renk körü birinin gözüyle bakma imkanı tanıyacak 

import numpy as np
from PIL import Image

def simulate_deuteranopia(image_path):
    print(f"Simülasyon hazırlanıyor: {image_path}...")
    
    # 1. Resmi Yükle
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Resmi sayı dizisine çevir (Matris)
    rgb_array = np.array(img, dtype=float)

    # --- BİLİMSEL MATRİSLER (Viénot et al. 1999) ---
    
    # RGB'den LMS'ye (Göz konilerine) geçiş
    rgb_to_lms = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])

    # SİMÜLASYON MATRİSİ (Deuteranopia - Yeşil Körlüğü)
    # Ortadaki satır (Yeşil) bozulmuş ve diğerlerinden türetilmiştir.
    lms_blind_matrix = np.array([
        [1, 0, 0],              
        [0.494207, 0, 1.24827], 
        [0, 0, 1]               
    ])

    
    try:
        lms_to_rgb = np.linalg.inv(rgb_to_lms)
    except np.linalg.LinAlgError:
        print("Matris hatası! Numpy sürümünü kontrol et.")
        return

    # --- İŞLEM PİPELINE'I ---
    
    # 1. Adım: Göz diline çevir (LMS)
    lms_image = np.dot(rgb_array, rgb_to_lms.T)

    # 2. Adım: Körlüğü uygula (Yeşili boz)
    simulated_lms = np.dot(lms_image, lms_blind_matrix.T)

    # 3. Adım: Tekrar ekrana çevir (RGB)
    final_rgb = np.dot(simulated_lms, lms_to_rgb.T)

    # Renkleri düzelt (0-255 arasına sıkıştır)
    final_rgb = np.clip(final_rgb, 0, 255).astype('uint8')

    # Göster ve Kaydet
    result_img = Image.fromarray(final_rgb)
    result_img.show()
    
    # Dosya adını ayarla 
    save_name = image_path.replace(".jpg", "_simulasyon.jpg").replace(".png", "_simulasyon.jpg")
    result_img.save(save_name)
    print(f"✅ Simülasyon bitti! Şu dosya oluşturuldu: {save_name}")

# --- TEST KISMI ---
if __name__ == "__main__":
    # İlk Deneme :Orijinal resim renk körüne nasıl görünüyor?
   #simulate_deuteranopia("kmeans_result_k4.jpg")
    
    # İkinci deneme :Yaptığı (K-Means) resim nasıl görünüyor?
    # (Eğer önceki koddan çıkan dosyanın adı farklıysa burayı düzelt)

   # simulate_deuteranopia("kmeans_result_k4.jpg")
   simulate_deuteranopia("corrected_output.jpg")