# app_utama.py
import streamlit as st

# Panggil st.set_page_config() SEBAGAI PERINTAH PERTAMA di skrip utama
# Ini mengatur judul tab browser, ikon, layout halaman, dan state awal sidebar
st.set_page_config(
    page_title="Deteksi Kematangan Cabai dengan Fuzzy dan Deteksi Hama dengan CNN",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded" # Sidebar langsung terbuka saat aplikasi dimuat
)

# --- Konten Halaman Utama ---

# Menampilkan pesan di sidebar untuk memandu pengguna
st.sidebar.success("Pilih halaman analisis dari menu navigasi di atas.")

# Judul Utama Aplikasi
st.write("# Selamat Datang di Aplikasi Deteksi Kematangan Cabai dengan Fuzzy dan Deteksi Hama dengan CNN")

# Penjelasan singkat dan panduan awal
st.write("Aplikasi ini dirancang untuk membantu Anda menganalisis kondisi tanaman cabai menggunakan Yolo V8")
st.info("Silakan pilih jenis analisis yang ingin Anda lakukan dari menu navigasi di sidebar sebelah kiri.")
st.markdown("---") # Garis pemisah

# --- Deskripsi Rinci Aplikasi ---
st.markdown("## Tentang Aplikasi Ini")

# A. Fungsi Aplikasi
st.subheader("Fungsi Aplikasi Ini")
st.markdown("""
Aplikasi web Streamlit ini berfungsi sebagai **Dashboard Analisis Tanaman Cabai Terintegrasi**. Tujuan utamanya adalah menyediakan antarmuka yang mudah digunakan bagi pengguna (seperti petani, peneliti, atau penghobi) untuk:

* **Menganalisis Kondisi Visual Tanaman Cabai:** Menggunakan model AI untuk memeriksa gambar tanaman cabai.
* **Mendeteksi Tingkat Kematangan & Kondisi Buah:** Menggunakan model **YOLOv8** untuk mengidentifikasi buah cabai pada gambar dan mengklasifikasikannya ke dalam kategori seperti 'Matang', 'Mentah', 'Sakit', 'Busuk', atau 'Kerusakan Fisik'.
* **Memberikan Rekomendasi Panen:** Menggunakan **Logika Fuzzy** yang mengolah hasil deteksi kematangan/kondisi dari YOLOv8 (dalam bentuk persentase) untuk memberikan saran tindakan panen (Tunda Panen, Panen Sebagian, Panen Penuh).
* **Mendeteksi Keberadaan Hama:** Menggunakan model **CNN (Convolutional Neural Network)** yang telah dilatih untuk mengidentifikasi jenis hama umum ('Kutu Daun', 'Kutu Kebul', 'Lalat Buah', dll.) yang mungkin ada pada gambar tanaman.
* **Input Fleksibel:** Memungkinkan analisis gambar melalui **Unggah File** atau **Ambil Gambar Langsung dari Kamera**.
* **Kontrol Perangkat Keras (Placeholder):** Menyediakan tombol ON/OFF sederhana yang terhubung ke **Firebase Realtime Database** untuk mengirim perintah kontrol ke sistem penyiraman (jika diaktifkan dan dikonfigurasi).

Secara keseluruhan, aplikasi ini bertujuan untuk membantu pengambilan keputusan dalam manajemen tanaman cabai dengan memanfaatkan teknologi AI dan logika fuzzy.
""")

# B. Cara Penggunaan
st.subheader("Cara Penggunaan")
st.markdown("""
1.  **Persiapan:**
    * Pastikan semua dependensi Python (**streamlit, ultralytics, tensorflow, keras, scikit-fuzzy, opencv-python, Pillow, numpy, firebase-admin**) sudah terinstal.
    * Pastikan path ke file model **YOLOv8 (.pt)** dan **CNN (.keras/.h5)**, serta file **kredensial Firebase (.json)** dan **URL Realtime Database** sudah diatur dengan benar di dalam kode skrip halaman (`1_Kematangan_Fuzzy.py` dan `2_Deteksi_Hama.py`).
2.  **Menjalankan Aplikasi:** Buka terminal, navigasi ke folder proyek Anda (yang berisi `app_utama.py` dan folder `pages`), lalu jalankan perintah: `streamlit run app_utama.py`.
3.  **Navigasi:** Gunakan menu di **sidebar kiri** yang secara otomatis dibuat oleh Streamlit untuk memilih halaman analisis: **"1 Kematangan Fuzzy"** atau **"2 Deteksi Hama"**.
4.  **Pilih Mode Input:** Di sidebar pada halaman aktif, pilih mode **"Unggah Gambar"** atau **"Ambil Gambar Kamera"**.
5.  **Sediakan Gambar:** Unggah file gambar Anda atau ambil foto menggunakan widget yang muncul di area utama. Gambar akan ditampilkan sebagai pratinjau.
6.  **Atur Parameter (Opsional):** Di sidebar, sesuaikan nilai **Threshold Kepercayaan** untuk model YOLO atau CNN jika Anda ingin mengubah sensitivitas deteksi.
7.  **Analisis:** Klik tombol **"Analisis..."** yang tersedia di bawah gambar input pada halaman utama.
8.  **Lihat Hasil:** Tunggu beberapa saat hingga proses selesai. Hasil analisis (gambar beranotasi, jumlah deteksi, persentase, rekomendasi/prediksi) akan ditampilkan di area utama halaman.
9.  **Kontrol Firebase:** Jika fitur Firebase aktif dan terkonfigurasi, gunakan tombol **"ON" / "OFF"** di sidebar untuk mengirim perintah kontrol penyiraman.
""")

# C. Detail Halaman 1: Deteksi Kematangan & Fuzzy
st.subheader("Detail Fungsi: Halaman 1 - Kematangan & Fuzzy")
st.markdown("""
* **Fokus:** Menganalisis kondisi dan tingkat kematangan buah cabai pada gambar, lalu memberikan rekomendasi panen.
* **Alur Kerja:**
    1.  Model **YOLOv8** mendeteksi & mengklasifikasikan setiap cabai ('Sakit', 'Matang', dll.).
    2.  Jumlah deteksi per kelas dihitung & gambar anotasi ditampilkan.
    3.  Persentase tiap kondisi (terutama % Matang, % Mentah, % Bermasalah) dihitung.
    4.  Persentase tersebut menjadi input untuk **Logika Fuzzy**.
    5.  Sistem Fuzzy menghasilkan **rekomendasi panen** (Tunda/Sebagian/Penuh).
""")

# D. Detail Halaman 2: Deteksi Hama CNN
st.subheader("Detail Fungsi: Halaman 2 - Deteksi Hama (CNN)")
st.markdown("""
* **Fokus:** Mengidentifikasi jenis hama utama yang mungkin ada pada gambar tanaman/daun cabai.
* **Alur Kerja:**
    1.  Gambar input di-**preprocessing** (resize & normalisasi).
    2.  Gambar dimasukkan ke model **CNN** yang sudah dilatih.
    3.  Model CNN mengeluarkan **probabilitas** untuk setiap kelas hama.
    4.  Kelas hama dengan probabilitas tertinggi ditampilkan sebagai **prediksi hama**
""")

st.markdown("---")
