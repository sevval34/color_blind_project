import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. PARÇA: RENK KÖRLÜĞÜ SİMÜLASYONU (Kişi 1'in Mantığı)
# ---------------------------------------------------------
def simulate_color_blindness(input_img):
    # DÜZELTME: Resmi zorla RGB yap (Şeffaflığı at)
    img = input_img.convert('RGB')
    rgb_array = np.array(img, dtype=float)

    # RGB -> LMS Matrisi
    rgb_to_lms = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])
    
    # Yeşil Körlüğü (Deuteranopia) Matrisi
    lms_blind_matrix = np.array([
        [1, 0, 0],              
        [0.494207, 0, 1.24827], 
        [0, 0, 1]               
    ])

    # Ters Çevirme Matrisi
    try:
        lms_to_rgb = np.linalg.inv(rgb_to_lms)
    except:
        return input_img 

    # İşlemler
    lms_image = np.dot(rgb_array, rgb_to_lms.T)
    simulated_lms = np.dot(lms_image, lms_blind_matrix.T)
    final_rgb = np.dot(simulated_lms, lms_to_rgb.T)
    
    # Görüntüyü toparla
    final_rgb = np.clip(final_rgb, 0, 255).astype('uint8')
    return Image.fromarray(final_rgb)

# ---------------------------------------------------------
# 2. PARÇA: YAPAY ZEKA K-MEANS (Kişi 2'nin Mantığı)
# ---------------------------------------------------------
def apply_kmeans(input_img, k=8):
    # DÜZELTME: Resmi zorla RGB yap (HATAYI ÇÖZEN SATIR BURASI)
    input_img = input_img.convert('RGB')

    # Resmi biraz küçült (Hız için)
    original_w, original_h = input_img.size
    img_small = input_img.resize((200, 200)) 
    
    image_array = np.array(img_small)
    w, h, d = image_array.shape
    
    # Resim verisini düzleştir
    pixel_list = image_array.reshape(-1, 3)

    # AI Modeli
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(pixel_list)
    
    # Renkleri değiştir
    new_colors = kmeans.cluster_centers_.astype('uint8')
    labels = kmeans.labels_
    quantized_pixels = new_colors[labels]
    
    quantized_image = quantized_pixels.reshape((h, w, 3))
    result_img = Image.fromarray(quantized_image)
    
    # Orijinal boyuta geri büyüt
    result_img = result_img.resize((original_w, original_h), Image.NEAREST)
    return result_img

# ---------------------------------------------------------
# 3. PARÇA: SENİN DÜZELTME KODUN (Kişi 3'ün Mantığı)
# ---------------------------------------------------------
def apply_correction_logic(input_img):
    # DÜZELTME: Girişin RGB olduğundan emin ol
    input_img = input_img.convert('RGB')
    
    # PIL -> OpenCV (RGB -> BGR) dönüşümü
    img_np = np.array(input_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # HSV Uzayına geç
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Maskeleme (Kırmızı ve Yeşil)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Müdahale: Kırmızıyı Parlat (+60), Yeşili Karart (-60)
    hsv[:, :, 2] = np.where(mask_red == 255, np.minimum(hsv[:, :, 2] + 60, 255), hsv[:, :, 2])
    hsv[:, :, 2] = np.where(mask_green == 255, np.maximum(hsv[:, :, 2] - 60, 0), hsv[:, :, 2])

    # Geri Dönüş: HSV -> BGR -> RGB -> PIL
    corrected_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(corrected_rgb)

# ---------------------------------------------------------
# 4. PARÇA: WEB ARAYÜZÜ (Kişi 4 - ENTEGRASYON)
# ---------------------------------------------------------
st.set_page_config(page_title="Renk Körü Asistanı", layout="wide")

st.title("👁️ Yapay Zeka Destekli Renk Körlüğü Asistanı")
st.markdown("""
Bu proje, **renk körü bireylerin** ayırt etmekte zorlandığı renkleri (Örn: Kırmızı/Yeşil),
Yapay Zeka (K-Means) ve Görüntü İşleme teknikleri kullanarak ayrıştırır.
""")

uploaded_file = st.file_uploader("Lütfen bir resim yükleyin...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # 1. Satır: Mevcut Durum
    st.subheader("1. Mevcut Durum Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Orijinal Görüntü", use_container_width=True)
    
    with col2:
        with st.spinner('Simülasyon hesaplanıyor...'):
            simulated_original = simulate_color_blindness(original_image)
            st.image(simulated_original, caption="Renk Körü Birinin Gördüğü", use_container_width=True)

    if st.button("🚀 Görüntüyü İyileştir ve Çözüm Üret"):
        st.divider()
        st.subheader("2. Yapay Zeka ve İyileştirme Süreci")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.info("Adım 1: K-Means Sadeleştirme")
            with st.spinner('AI renkleri grupluyor...'):
                kmeans_img = apply_kmeans(original_image, k=8)
                st.image(kmeans_img, caption="AI Sadeleştirdi", use_container_width=True)
        
        with col4:
            st.success("Adım 2: Kontrast Ayrıştırma")
            with st.spinner('Renkler ayrıştırılıyor...'):
                corrected_img = apply_correction_logic(kmeans_img)
                st.image(corrected_img, caption="İyileştirilmiş Görüntü", use_container_width=True)
        
        with col5:
            st.error("Adım 3: Sonuç Testi")
            with st.spinner('Son kontrol yapılıyor...'):
                final_simulation = simulate_color_blindness(corrected_img)
                st.image(final_simulation, caption="Renk Körü İçin Sonuç", use_container_width=True)
        
        st.balloons()
        st.success("✅ İŞLEM TAMAMLANDI!")