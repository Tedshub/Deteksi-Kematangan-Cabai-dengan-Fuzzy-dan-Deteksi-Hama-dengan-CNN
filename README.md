# Deteksi Kematangan Cabai dan Deteksi Hama

## Deskripsi Proyek
Proyek ini merupakan sistem berbasis computer vision yang dirancang untuk:
1. Mendeteksi tingkat kematangan cabai menggunakan metode **Fuzzy Logic**.
2. Mendeteksi keberadaan hama pada tanaman cabai menggunakan model **Convolutional Neural Network (CNN)**.

Aplikasi ini dibangun menggunakan **Streamlit** sebagai antarmuka web interaktif, sehingga dapat dijalankan langsung melalui browser.

---

## Teknologi yang Digunakan
- Python 3.10+
- Streamlit
- Ultralytics (YOLO)
- TensorFlow
- scikit-fuzzy
- OpenCV
- NumPy
- Pillow
- Firebase Admin
- Streamlit Option Menu

---

## Struktur Folder
```
STREAMLIT_DETEKSI_CABAI_DAN_HAMA/
│
├── .idea/                                   # Konfigurasi project untuk IDE (misal PyCharm)
│   ├── inspectionProfiles/
│   ├── misc.xml
│   ├── modules.xml
│   ├── Streamlit_Deteksi_Cabai_dan_Hama.iml
│   ├── vcs.xml
│   └── workspace.xml
│
├── credentials/
│   └── iot-kematangancabai-hama-firebase-adminsdk-xxxx.json   # Kredensial Firebase
│
├── model_hama/
│   └── hama_cabai_MobileNetV2_final_e38.weights.h5             # Model CNN untuk deteksi hama
│
├── pages/
│   ├── 1_kematangan_fuzzy.py                                   # Modul deteksi kematangan cabai (Fuzzy)
│   ├── 2_deteksi_hama.py                                       # Modul deteksi hama (CNN)
│   └── 3_Alat_siram_dan_pupuk.py                               # Modul kontrol alat siram & pupuk
│
├── venv/                                                       # Virtual environment (tidak di-push ke repo)
│
├── yolo_weight_cabai/
│   └── best.pt                                                 # Model YOLO untuk deteksi objek cabai
│
├── .gitignore                                                  # File untuk mengecualikan file/folder tertentu dari Git
├── final_train.ipynb                                           # Notebook pelatihan model
├── main.py                                                     # File utama Streamlit
├── README.md                                                   # Dokumentasi proyek
└── requirements.txt                                            # Daftar dependensi Python
```

---

## Cara Menjalankan Proyek

### 1. Clone Repository
```bash
git clone https://github.com/username/DeteksiHamaCabai.git
cd DeteksiHamaCabai
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
```

### 3. Aktifkan Virtual Environment
#### Windows
```bash
.env\Scriptsctivate
```
#### macOS / Linux
```bash
source venv/bin/activate
```

### 4. Install Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi
```bash
streamlit run main.py
```

Setelah dijalankan, aplikasi dapat diakses melalui browser pada alamat:
```
http://localhost:8501
```

---

## requirements.txt
```
streamlit
ultralytics
numpy
scikit-fuzzy
opencv-python
Pillow
firebase-admin
tensorflow
streamlit-option-menu
```

---

## Fitur Utama
- Deteksi tingkat kematangan cabai menggunakan logika fuzzy.
- Deteksi hama pada daun cabai dengan model CNN.
- Antarmuka web interaktif berbasis Streamlit.
- Visualisasi hasil prediksi secara real-time.

---

## Lisensi
Proyek ini dilisensikan di bawah lisensi MIT. Anda bebas untuk menggunakan, memodifikasi, dan mengembangkan proyek ini untuk kepentingan penelitian dan pendidikan.
