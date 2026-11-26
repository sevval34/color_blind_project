import numpy as np
from PIL import Image

#resmi açtı
img=Image.open("images/original.jpg")
#resmi numpy dizisine çevir
rgb_data=np.array(img,dtype=float)

#RGB->LS dönüşümü
transformation_matrix = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
])

# numpy.dot fonksiyonu, 1.4 milyon pikseli tek seferde çarpar.
lms_data = np.dot(rgb_data, transformation_matrix.T)

# 5. Sonuçları Kontrol Et
print("--- DÖNÜŞÜM RAPORU ---")
print(f"Orijinal RGB boyutu: {rgb_data.shape}")
print(f"Dönüştürülen LMS boyutu: {lms_data.shape}")
print("-" * 30)
print(f"Sol üst piksel (RGB): {rgb_data[0,0]}") 
print(f"Sol üst piksel (LMS): {lms_data[0,0]}") 
print("-" * 30)
print("✅ Tebrikler! Görüntü artık insan gözü (LMS) uzayında.")