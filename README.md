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
DeteksiHamaCabai/
│
├── venv/                      # Virtual environment
├── main.py                    # File utama aplikasi Streamlit
├── requirements.txt           # Daftar dependensi
├── models/                    # Model CNN dan konfigurasi fuzzy
├── data/                      # Dataset pelatihan dan pengujian
├── utils/                     # Fungsi bantu untuk deteksi dan preprocessing
└── README.md
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