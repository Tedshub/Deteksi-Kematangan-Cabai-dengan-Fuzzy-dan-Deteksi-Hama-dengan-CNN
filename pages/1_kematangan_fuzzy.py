# pages/1_Kematangan_Fuzzy.py

import streamlit as st
from ultralytics import YOLO
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import cv2
from PIL import Image
import io
from pathlib import Path
import time
import firebase_admin
from firebase_admin import credentials, db
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI (Perlu disesuaikan di setiap page atau pakai file config terpisah)
# ==============================================================================
MODEL_YOLO_PATH = Path('yolo_weight_cabai/best.pt') # <-- UBAH INI! Sesuaikan path relatif/absolut
FIREBASE_CRED_PATH = Path('credentials/iot-kematangancabai-hama-firebase-adminsdk-fbsvc-db0022ef3f.json') # <-- UBAH INI!
FIREBASE_DB_URL = 'https://console.firebase.google.com/u/0/project/iot-kematangancabai-hama/database/iot-kematangancabai-hama-default-rtdb/data/~2F' # <-- UBAH INI!

CLASS_NAMES_YOLO = {
    0 : 'Sakit', 1 : 'Kerusakan Fisik', 2 : 'Matang', 3 : 'Busuk', 4 : 'Mentah'
}
INDEX_SAKIT = 0; INDEX_RUSAK = 1; INDEX_MATANG = 2; INDEX_BUSUK = 3; INDEX_MENTAH = 4
DEFAULT_CONF_YOLO = 0.10

# ==============================================================================
# INISIALISASI FIREBASE & FUNGSI HELPER (Bisa juga ditaruh di utils.py)
# ==============================================================================
@st.cache_resource
def initialize_firebase(cred_path, db_url):
    # ... (Kode fungsi initialize_firebase sama seperti sebelumnya) ...
    if not cred_path.is_file():
        # st.sidebar.warning(f"File Kredensial Firebase tidak ditemukan: {cred_path}.") # Komentari agar tidak muncul di setiap page jika tidak perlu
        return None, False
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred, {'databaseURL': db_url})
            print("Firebase Initialized!") # Log ke konsol saja
            return db, True
        else:
            return db, True
    except Exception as e:
        st.sidebar.error(f"Firebase Init Error: {e}")
        return None, False

db_ref, firebase_ready = initialize_firebase(FIREBASE_CRED_PATH, FIREBASE_DB_URL)

@st.cache_resource
def load_yolo_model(model_path_str):
    # ... (Kode fungsi load_model untuk yolo sama seperti sebelumnya, tanpa st.success/error agar tidak berulang) ...
    model_path = Path(model_path_str)
    if not model_path.is_file(): return None
    try:
        model = YOLO(str(model_path))
        print(f"Model YOLO dimuat dari: {model_path}") # Log ke konsol
        return model
    except Exception as e:
        print(f"Error memuat model YOLO: {e}")
        return None

# --- Fungsi Fuzzy Logic (Sama seperti sebelumnya) ---
def define_fuzzy_system():
    # ... (Kode definisi variabel dan MF fuzzy sama) ...
    persen_matang = ctrl.Antecedent(np.arange(0, 101, 1), 'persen_matang')
    persen_mentah = ctrl.Antecedent(np.arange(0, 101, 1), 'persen_mentah')
    persen_bermasalah = ctrl.Antecedent(np.arange(0, 101, 1), 'persen_bermasalah')
    keputusan_panen = ctrl.Consequent(np.arange(0, 11, 1), 'keputusan_panen', defuzzify_method='centroid')
    persen_matang['rendah'] = fuzz.trimf(persen_matang.universe, [0, 0, 40]); persen_matang['sedang'] = fuzz.trimf(persen_matang.universe, [30, 55, 80]); persen_matang['tinggi'] = fuzz.trimf(persen_matang.universe, [70, 100, 100])
    persen_mentah['rendah'] = fuzz.trimf(persen_mentah.universe, [0, 0, 35]); persen_mentah['sedang'] = fuzz.trimf(persen_mentah.universe, [25, 50, 75]); persen_mentah['tinggi'] = fuzz.trimf(persen_mentah.universe, [65, 100, 100])
    persen_bermasalah['rendah'] = fuzz.trimf(persen_bermasalah.universe, [0, 0, 15]); persen_bermasalah['sedang'] = fuzz.trimf(persen_bermasalah.universe, [10, 25, 40]); persen_bermasalah['tinggi'] = fuzz.trapmf(persen_bermasalah.universe, [35, 50, 100, 100])
    keputusan_panen['tunda'] = fuzz.trimf(keputusan_panen.universe, [0, 2, 4]); keputusan_panen['sebagian'] = fuzz.trimf(keputusan_panen.universe, [3, 5, 7]); keputusan_panen['penuh'] = fuzz.trimf(keputusan_panen.universe, [6, 8, 10])
    return persen_matang, persen_mentah, persen_bermasalah, keputusan_panen

def run_fuzzy_logic(input_persen_matang, input_persen_mentah, input_persen_bermasalah):
    # ... (Kode eksekusi fuzzy logic dan aturan sama seperti sebelumnya) ...
    persen_matang, persen_mentah, persen_bermasalah, keputusan_panen = define_fuzzy_system()
    rule1 = ctrl.Rule(persen_bermasalah['tinggi'], keputusan_panen['tunda']); rule2 = ctrl.Rule(persen_mentah['tinggi'] & (persen_bermasalah['rendah'] | persen_bermasalah['sedang']), keputusan_panen['tunda']); rule3 = ctrl.Rule(persen_matang['tinggi'] & persen_mentah['rendah'] & persen_bermasalah['rendah'], keputusan_panen['penuh']); rule4 = ctrl.Rule(persen_matang['sedang'] & persen_bermasalah['rendah'], keputusan_panen['sebagian']); rule5 = ctrl.Rule(persen_matang['tinggi'] & (persen_mentah['sedang'] | persen_bermasalah['sedang']), keputusan_panen['sebagian']); rule6 = ctrl.Rule(persen_matang['sedang'] & persen_bermasalah['sedang'], keputusan_panen['sebagian']); rule7 = ctrl.Rule(persen_matang['rendah'] & persen_bermasalah['rendah'], keputusan_panen['tunda'])
    harvesting_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7]); harvesting_simulation = ctrl.ControlSystemSimulation(harvesting_ctrl)
    harvesting_simulation.input['persen_matang'] = input_persen_matang; harvesting_simulation.input['persen_mentah'] = input_persen_mentah; harvesting_simulation.input['persen_bermasalah'] = input_persen_bermasalah
    try:
        harvesting_simulation.compute(); hasil_crisp = harvesting_simulation.output['keputusan_panen']
        rekomendasi = "";
        if hasil_crisp <= 4.0: rekomendasi = "TUNDA PANEN (Periksa Kondisi)"
        elif hasil_crisp <= 7.0: rekomendasi = "PANEN SEBAGIAN (Prioritaskan Sehat & Matang)"
        else: rekomendasi = "PANEN PENUH (Kondisi Ideal)"
        return hasil_crisp, rekomendasi
    except ValueError: return None, "TIDAK DAPAT DITENTUKAN";
    except Exception as e: return None, f"ERROR FUZZY: {e}"


# --- Fungsi Update Firebase (Sama seperti sebelumnya) ---
def update_firebase_state(db_conn, state_value):
    # ... (Kode fungsi update_firebase_state sama seperti sebelumnya, menggunakan path 'kontrol/penyiraman/status') ...
    state_node_path = 'kontrol/penyiraman/status';
    if db_conn and firebase_ready:
        try: ref = db_conn.reference(state_node_path); ref.set(state_value); st.sidebar.success(f"Perintah {'ON' if state_value else 'OFF'} terkirim!"); time.sleep(0.5); return True # Kurangi delay
        except Exception as e: st.sidebar.error(f"Firebase Error: {e}"); return False
    else: st.sidebar.warning("Firebase tidak siap."); return False

# ==============================================================================
# STREAMLIT UI & LOGIC (Halaman Kematangan & Fuzzy)
# ==============================================================================

st.title("1️⃣ Deteksi Kematangan/Kondisi & Rekomendasi Panen")
st.write("Analisis kondisi cabai menggunakan YOLOv8 dan dapatkan rekomendasi panen.")

# --- Muat Model YOLO ---
# Gunakan string path karena @st.cache_resource tidak bisa handle Path object kadang2
model_yolo = load_yolo_model(str(MODEL_YOLO_PATH))

# --- Sidebar Khusus Halaman Ini ---
st.sidebar.header("Pengaturan Halaman Ini")
mode_kematangan = st.sidebar.radio("Mode Input Kematangan:", ("Unggah Gambar", "Ambil Gambar Kamera"), key="mode_kematangan")
conf_yolo = st.sidebar.slider("Threshold Kepercayaan YOLO (%)", 10, 95, int(DEFAULT_CONF_YOLO*100), 5, key="conf_yolo_kematangan") / 100.0

# --- Area Utama ---
image_to_analyze_pil = None
source_info = ""

if mode_kematangan == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih Gambar Cabai...", type=["jpg", "jpeg", "png"], key="uploader_kematangan")
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            image_to_analyze_pil = Image.open(io.BytesIO(bytes_data))
            source_info = f"Gambar Unggahan: {uploaded_file.name}"
        except Exception as e: st.error(f"Gagal memuat gambar: {e}")
elif mode_kematangan == "Ambil Gambar Kamera":
    img_file_buffer = st.camera_input("Arahkan kamera dan ambil gambar:", key="camera_kematangan")
    if img_file_buffer is not None:
        try:
            bytes_data = img_file_buffer.getvalue()
            image_to_analyze_pil = Image.open(io.BytesIO(bytes_data))
            source_info = "Gambar dari Kamera"
        except Exception as e: st.error(f"Gagal memproses gambar kamera: {e}")

# Tampilkan gambar input
if image_to_analyze_pil:
    st.subheader("Gambar Input")
    st.image(image_to_analyze_pil, caption=source_info, use_column_width=False, width=200)

    # Tombol Analisis
    if st.button(f"Analisis Kondisi & Kematangan", key="analyze_kematangan"):
        if model_yolo: # Cek model sudah termuat
            st.markdown("---")
            st.subheader("🔍 Hasil Analisis YOLOv8 & Fuzzy Logic")
            start_analysis_time = time.time()
            with st.spinner('Menganalisis kondisi/kematangan...'):
                try:
                    # 1. Deteksi YOLO
                    results_yolo = model_yolo.predict(source=image_to_analyze_pil, conf=conf_yolo, verbose=False)
                    annotated_yolo_img_np = None
                    jumlah_cabai_terdeteksi_per_kelas = {idx: 0 for idx in CLASS_NAMES_YOLO.keys()}
                    total_cabai_terdeteksi = 0

                    if results_yolo and len(results_yolo) > 0 and results_yolo[0].boxes:
                        boxes = results_yolo[0].boxes; total_cabai_terdeteksi = len(boxes)
                        annotated_yolo_img_np = results_yolo[0].plot(conf=True)
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            if cls_id in jumlah_cabai_terdeteksi_per_kelas: jumlah_cabai_terdeteksi_per_kelas[cls_id] += 1
                        st.info(f"YOLO: {total_cabai_terdeteksi} objek terdeteksi.")
                    else: st.info("YOLO: Tidak ada objek kondisi/kematangan terdeteksi.")

                    # Tampilkan hasil YOLO
                    if annotated_yolo_img_np is not None:
                        st.image(annotated_yolo_img_np, caption='Deteksi Kondisi/Kematangan YOLOv8', channels="BGR", use_column_width=True, width=200)

                    st.write("**Jumlah Deteksi:**")
                    if total_cabai_terdeteksi > 0:
                        # Tampilkan metrik
                        rows = len(CLASS_NAMES_YOLO) // 3 + (1 if len(CLASS_NAMES_YOLO) % 3 else 0); idx = 0
                        for r in range(rows):
                             metric_cols = st.columns(3)
                             for c in range(3):
                                 if idx < len(CLASS_NAMES_YOLO):
                                     with metric_cols[c]: st.metric(label=f"{CLASS_NAMES_YOLO[idx]}", value=jumlah_cabai_terdeteksi_per_kelas.get(idx,0))
                                     idx += 1

                        # 2. Hitung Persentase & Jalankan Fuzzy
                        input_persen_matang = (jumlah_cabai_terdeteksi_per_kelas.get(INDEX_MATANG, 0) / total_cabai_terdeteksi) * 100
                        input_persen_mentah = (jumlah_cabai_terdeteksi_per_kelas.get(INDEX_MENTAH, 0) / total_cabai_terdeteksi) * 100
                        input_persen_bermasalah = ((jumlah_cabai_terdeteksi_per_kelas.get(INDEX_SAKIT, 0) + jumlah_cabai_terdeteksi_per_kelas.get(INDEX_RUSAK, 0) + jumlah_cabai_terdeteksi_per_kelas.get(INDEX_BUSUK, 0)) / total_cabai_terdeteksi) * 100

                        st.write("**Input Fuzzy Logic:**")
                        scol1, scol2, scol3 = st.columns(3)
                        scol1.metric("Matang", f"{input_persen_matang:.1f}%"); scol2.metric("Mentah", f"{input_persen_mentah:.1f}%"); scol3.metric("Bermasalah", f"{input_persen_bermasalah:.1f}%")

                        fuzzy_score, fuzzy_recommendation = run_fuzzy_logic(input_persen_matang, input_persen_mentah, input_persen_bermasalah)

                        st.write("**Rekomendasi Panen:**")
                        if fuzzy_score is not None: st.metric("Skor Fuzzy (0-10)", f"{fuzzy_score:.2f}")
                        if "TUNDA" in fuzzy_recommendation: st.warning(f"**{fuzzy_recommendation}**")
                        elif "SEBAGIAN" in fuzzy_recommendation: st.info(f"**{fuzzy_recommendation}**")
                        elif "PENUH" in fuzzy_recommendation: st.success(f"**{fuzzy_recommendation}**")
                        else: st.error(f"**{fuzzy_recommendation}**")
                    else:
                        st.info("Tidak dapat menjalankan Fuzzy Logic (tidak ada deteksi).")

                except Exception as e:
                    st.error(f"Terjadi error saat analisis: {e}")
                    import traceback; st.error(traceback.format_exc())

            end_analysis_time = time.time()
            st.caption(f"Analisis selesai dalam {end_analysis_time - start_analysis_time:.2f} detik.")
        else:
            st.warning("Model YOLO belum siap. Periksa path model.")

else:
     st.info("Silakan unggah gambar atau ambil gambar dari kamera untuk memulai analisis.")