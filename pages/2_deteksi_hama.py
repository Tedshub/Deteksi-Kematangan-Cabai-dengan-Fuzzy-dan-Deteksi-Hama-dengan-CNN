import streamlit as st
import tensorflow as tf
# load_model tidak diperlukan lagi
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2 # Import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input # Tambahkan Input layer
from tensorflow.keras.models import Model # Diperlukan untuk membuat model
# Import fungsi preprocess_input spesifik untuk MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
import numpy as np
from pathlib import Path
import time # Tambahkan import time


# --- Parameter Model (SPESIFIK UNTUK MOBILENETV2 WEIGHTS DARI TRAINING SCRIPT) ---
# Path relatif dari file Streamlit ke folder model_hama
# Sesuaikan jika struktur folder Anda berbeda
MODEL_DIR = Path(__file__).resolve().parent.parent / "model_hama"


NAMA_MODEL_USED = 'MobileNetV2' # Sesuai dengan NAMA_MODEL di script training

MODEL_WEIGHTS_FILENAME = f'hama_cabai_{NAMA_MODEL_USED}_final_e38.weights.h5' # <--- SESUAIKAN NAMA FILE WEIGHTS DI SINI!


MODEL_WEIGHTS_PATH = MODEL_DIR / MODEL_WEIGHTS_FILENAME

# Parameter gambar (harus sama dengan saat training)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) # Input awal model Keras menerima gambar 0-255

# Daftar Nama Kelas (WAJIB DISESUAIKAN DAN URUTANNYA HARUS BENAR!)
# Pastikan ini SAMA PERSIS dengan CLASS_NAMES_PEST di script training
CLASS_NAMES = ['kutu_daun', 'kutu_kebul', 'lalat_buah', 'thrips', 'tungau', 'ulat_grayak', 'ulat_penggerek_buah']
NUM_CLASSES = len(CLASS_NAMES)

if not CLASS_NAMES:
    st.warning("Daftar CLASS_NAMES belum dikonfigurasi. Menggunakan placeholder.")
    # Ganti angka 7 dengan jumlah kelas Anda jika berbeda
    CLASS_NAMES = [f"Hama Kelas {i + 1}" for i in range(7)]
    NUM_CLASSES = len(CLASS_NAMES)
# --- Fungsi Helper ---

@st.cache_resource
def get_model(weights_path, input_shape, num_classes):
    """Mendefinisikan arsitektur MobileNetV2 (sesuai training script) dan memuat bobot."""
    st.info("Memuat model MobileNetV2...")
    try:
        # 1. Definisikan Input Layer (sesuai input_gambar_hama di training)
        inputs = Input(shape=input_shape, name="input_gambar_hama", dtype=tf.float32)

        # 2. Tambahkan layer preprocessing (sesuai dengan training script)
        # Layer Augmentasi ada di training tapi TIDAK AKTIF saat inference, jadi bisa diabaikan
        # Layer Preprocessing persis seperti di training
        x = mobilenet_v2_preprocess_input(inputs) # Input 0-255 menjadi -1 sampai 1

        # 3. Definisikan base model MobileNetV2
        base_model = MobileNetV2(input_tensor=x, # Sambungkan ke output preprocessing
                                 input_shape=input_shape, # input_shape di sini untuk konfigurasi base model
                                 include_top=False, # Jangan sertakan layer klasifikasi teratas
                                 weights=None) # Jangan load pretrained weights ImageNet

        # 4. Tambahkan Top Layers (Classifier) - HARUS SAMA PERSIS DENGAN TRAINING
        x_top = base_model.output
        x_top = GlobalAveragePooling2D(name="global_pooling")(x_top) # Layer Global Average Pooling
        outputs = Dense(num_classes, activation='softmax', name="output_prediksi")(x_top) # Layer output

        # 5. Gabungkan layer-layer menjadi model final
        model = Model(inputs=inputs, outputs=outputs, name=f"PestDetector_{NAMA_MODEL_USED}_Streamlit")

        # 6. Muat bobot dari file .weights.h5
        print(f"Mencari file bobot di: {weights_path}")
        if not Path(weights_path).is_file(): # Gunakan is_file() untuk memastikan itu file
            st.error(f"Error: File bobot tidak ditemukan di: {weights_path}")
            print(f"Error: File bobot tidak ditemukan di: {weights_path}")
            # sys.exit(1) # Jangan exit di Streamlit, kembalikan None
            return None

        model.load_weights(weights_path)
        print(f"Bobot model berhasil dimuat dari {weights_path}")
        st.success("Model MobileNetV2 siap digunakan!")
        return model

    except Exception as e:
        st.error(f"Error saat memuat bobot model dari '{weights_path.name}': {e}")
        print(f"Detail error pemuatan bobot model: {e}")
        import traceback
        print(f"Traceback error pemuatan bobot: {traceback.format_exc()}")
        return None

def preprocess_uploaded_image(pil_image, target_size):
    """
    Melakukan pra-pemrosesan pada gambar PIL yang diunggah (resize dan to_array).
    Preprocessing MobileNetV2 sekarang dilakukan DI DALAM model.
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img_resized = pil_image.resize(target_size)
    # Menggunakan tf.keras.utils.img_to_array menghasilkan array dengan dtype float32
    img_array = tf.keras.utils.img_to_array(img_resized)

    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch # Mengembalikan batch dengan nilai 0-255

def predict_and_return_results(model, processed_image_batch, class_names_list):
    """Melakukan prediksi dan mengembalikan hasil."""
    if model is None or processed_image_batch is None or not class_names_list:
        # Pesan error ditampilkan di UI utama
        return None, None # Kembalikan None jika ada masalah input awal

    try:
        # processed_image_batch yang masuk ke sini adalah 0-255,
        # model_loaded akan melakukan preprocessing -1 ke 1 di dalamnya
        start_pred_time = time.time() # Tambahkan pengukuran waktu prediksi
        predictions = model.predict(processed_image_batch)
        end_pred_time = time.time()

        score_softmax = predictions[0] # Asumsi output layer adalah softmax

        predicted_class_index = np.argmax(score_softmax)

        if predicted_class_index >= len(class_names_list):
             # Pesan error ditangani di UI
             return "Error: Indeks Kelas Prediksi tidak valid", 0.0 # Kembalikan pesan error spesifik

        predicted_class_name = class_names_list[predicted_class_index]
        confidence_score = 100 * np.max(score_softmax) # Max dari output softmax adalah confidence

        # Kembalikan nama kelas, confidence, dan waktu prediksi
        return predicted_class_name, confidence_score, (end_pred_time - start_pred_time)

    except Exception as e:
        import traceback
        print(f"Traceback error prediksi: {traceback.format_exc()}")
        # Kembalikan pesan error dan confidence 0, waktu None
        return f"Error Prediksi: {str(e)[:60]}...", 0.0, None


# --- Antarmuka Streamlit untuk halaman Deteksi Hama ---
st.title("🌿 Deteksi Hama pada Tanaman Cabai")
st.markdown("Unggah gambar hama untuk diklasifikasikan menggunakan model Convolutional Neural Network (CNN).")
st.markdown("---")

# --- Informasi Model yang Digunakan (Hardcoded untuk MobileNetV2 Weights) ---
st.sidebar.header("Informasi Model Digunakan")
st.sidebar.info(f"Model Aktif: MobileNetV2 (Bobot: {MODEL_WEIGHTS_FILENAME})")
st.sidebar.markdown(f"Arsitektur Dasar: MobileNetV2")
st.sidebar.markdown(f"Ukuran Input Gambar: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
st.sidebar.markdown(f"Jumlah Kelas Hama Dikenali: {NUM_CLASSES}")
st.sidebar.markdown("---")
st.sidebar.warning("Pastikan file bobot model ada di lokasi yang benar dan CLASS_NAMES sesuai dengan saat training.")


# Memuat model (Sekarang memanggil dengan path bobot)
# Ini akan dijalankan setiap kali script dijalankan, tapi @st.cache_resource akan mencegah pemuatan ulang bobot jika fungsi tidak berubah
model_loaded = get_model(MODEL_WEIGHTS_PATH, INPUT_SHAPE, NUM_CLASSES)

if model_loaded:
    col1, col2 = st.columns(2)

    # Widget untuk upload gambar di kolom pertama
    with col1:
        uploaded_file = st.file_uploader(
            "Pilih gambar hama untuk dianalisis:",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_mobilenetv2" # Key unik
        )

    if uploaded_file is not None:
        # Pastikan gambar di kolom 1
        with col1:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Gambar yang Diunggah", use_column_width=True)

            # Tombol klasifikasi di bawah gambar
            if st.button(f"🔍 Klasifikasikan dengan {NAMA_MODEL_USED}", type="primary", use_container_width=True, key=f"button_{NAMA_MODEL_USED}"):
                if pil_image: # Seharusnya selalu True jika uploaded_file is not None
                    with st.spinner(f"Memproses gambar dan melakukan prediksi..."):
                        # Pra-pemrosesan gambar (resize dan to_array 0-255)
                        processed_image = preprocess_uploaded_image(pil_image, IMAGE_SIZE)
                        # Panggil fungsi prediksi
                        # Fungsi prediksi sekarang mengembalikan 3 nilai
                        predicted_class, confidence, pred_duration = predict_and_return_results(
                            model_loaded,
                            processed_image, # processed_image di sini masih 0-255
                            CLASS_NAMES
                        )
                        # Tampilkan hasil di kolom kedua
                        with col2:
                            if predicted_class and confidence is not None:
                                if "Error" in predicted_class:
                                    st.error(predicted_class)
                                else:
                                    st.subheader("Hasil Prediksi:")
                                    st.success(f"Terdeteksi sebagai: {predicted_class}")
                                    st.info(f"Tingkat Kepercayaan: {confidence:.2f}%")
                                    if pred_duration is not None:
                                         st.text(f"Waktu Prediksi: {pred_duration:.4f} detik")
                            else: # Jika predict_and_return_results mengembalikan None, None
                                st.error("Gagal mendapatkan hasil prediksi. Periksa input atau model.")
    else: # Jika tidak ada file yang diunggah (uploaded_file is None)
        with col1: # Pesan di kolom pertama
            st.info("Silakan unggah gambar untuk dianalisis.")
        with col2: # Kolom kedua bisa dibiarkan kosong atau diberi placeholder
            st.write("") # Atau st.info("Kolom hasil akan muncul di sini.")

else: # Jika model_loaded adalah None (gagal dimuat)
    st.error(f"Aplikasi tidak dapat berjalan karena model '{MODEL_WEIGHTS_FILENAME}' gagal dimuat.")


st.markdown("---")
st.caption("Aplikasi Deteksi Hama")