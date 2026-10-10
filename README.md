# Prediksi Penjualan Mobil Toyota di Indonesia Menggunakan Time Series Multilayer Perceptron (MLP)
> **Project Ujian Tengah Semester (UTS) — Mata Kuliah Kecerdasan Buatan**  
> **Program Studi D3 Manajemen Informatika — PSDKU Kota Kediri — Politeknik Negeri Malang**  
> **Dosen Pengampu:** Abidatul Izzah, M.Kom  

---

## Ringkasan Project
Aplikasi web berbasis **Flask** yang menerapkan pemodelan deret waktu (*time series forecasting*) menggunakan algoritma **Multilayer Perceptron (MLP)** dengan arsitektur **10-15-1** untuk memprediksi volume penjualan bulanan mobil Toyota di Indonesia.

Project ini memenuhi seluruh kriteria topik **Prediksi Data (Poin 7 & 8)** soal UTS:
1. **Data Time Series:** Menggunakan 149 bulan data historis deret waktu dari Januari 2011 hingga Mei 2023.
2. **Sumber Dataset Resmi:** Bersumber dari **GAIKINDO** (*Gabungan Industri Kendaraan Bermotor Indonesia*) pada bagian *Indonesian Automobile Wholesales by Brand*.
3. **Preprocessing Data:** Normalisasi MinMaxScaler [0, 1] dan pembentukan fitur *Sliding Window / Lag-10*.
4. **Model Multilayer Perceptron (MLP):** Arsitektur 10 Input -> 15 Hidden (ReLU) -> 1 Output dengan Optimizer Adam.
5. **Evaluasi Model Sesuai Syarat:**
   * **MAE (*Mean Absolute Error*):** **1.947,71 Unit** (Rata-rata selisih tebakan sangat kecil dari rata-rata penjualan 27.000 unit/bulan).
   * **MSE (*Mean Squared Error*):** **7.754.258,95**.
   * **MAPE:** **7,86%** (Kategori peramalan: *Sangat Akurat* / di bawah 10%).
6. **Prediksi untuk Periode Tertentu:** Fitur pengujian manual bebas 10 bulan dengan visualisasi titik prediksi ke-11.
7. **Komparasi Artikel Jurnal (Poin 8 UTS):** Menggunakan artikel jurnal referensi resmi dari kampus sendiri:  
   *Muhammad Nabil Afkar, Dewi Titania Randa, Rizal Adi Saputra (November 2024). "Prediksi Penjualan Mobil Toyota di Indonesia Menggunakan Multi-Layer Perceptron"*, **Jurnal Informatika Polinema (JIP)**, Vol. 11 No. 1, Hal. 91–98. DOI: [10.33795/jip.v11i1.6339](https://doi.org/10.33795/jip.v11i1.6339).

---

## Struktur Direktori
```text
uts kb/
│
├── dataset/
│   └── penjualan_toyota_gaikindo.csv   # Data time series 149 bulan GAIKINDO (2011–2023)
│
├── models/
│   ├── mlp_time_series.pkl             # Model MLP terlatih & scaler
│   └── metrics.json                    # Metadata evaluasi MAE, MSE, & komparasi jurnal
│
├── static/
│   ├── css/
│   │   └── style.css                   # Desain Dark Automotive Sleek (Toyota Red & Slate Navy)
│   ├── js/
│   │   └── main.js                     # Integrasi Chart.js interaktif
│   └── img/
│       └── toyota_hero.jpg             # Foto otomotif Toyota
│
├── templates/
│   ├── base.html                       # Layout dasar navbar & footer (Logo SVG Toyota)
│   ├── index.html                      # Beranda & grafik tren 2011–2023
│   ├── dataset.html                    # Eksplorasi dataset GAIKINDO 149 baris
│   ├── evaluasi.html                   # Evaluasi MAE, MSE, & komparasi jurnal
│   └── prediksi.html                   # Uji simulasi peramalan input manual bebas
│
├── app.py                              # Server web Flask utama
├── train_model.py                      # Script pelatihan model MLP Time Series
├── wsgi.py                             # Entrypoint deploy PythonAnywhere
├── requirements.txt                    # Dependensi Python
├── .gitignore                          # Ignore cache/temporary files
└── README.md                           # Dokumentasi & panduan presentasi UTS
```

---

## Cara Menjalankan Secara Lokal

1. **Buka Terminal / Command Prompt** pada folder project ini:
   ```bash
   cd "e:\Semester 5\Kecerdasan Buatan\uts kb"
   ```

2. **Jalankan Aplikasi Flask:**
   ```bash
   python app.py
   ```

3. **Buka di Browser:**
   Akses `http://127.0.0.1:5000` pada browser Anda.

---

## Panduan Deploy ke PythonAnywhere (Via GitHub)

1. **Push ke GitHub:**
   ```bash
   git add .
   git commit -m "feat: complete Toyota sales time series MLP prediction"
   git push -u origin main
   ```

2. **Di Terminal Bash PythonAnywhere:**
   ```bash
   git clone https://github.com/USERNAME-ANDA/NAMA-REPO.git
   cd NAMA-REPO
   pip install -r requirements.txt
   ```

3. **Di Tab Web PythonAnywhere:**
   * Add a new web app -> Manual configuration -> Python 3.10.
   * Set Source code: `/home/USERNAME-ANDA/NAMA-REPO`
   * Buka file WSGI, isi dengan:
     ```python
     import sys
     import os

     project_home = '/home/USERNAME-ANDA/NAMA-REPO'
     if project_home not in sys.path:
         sys.path.insert(0, project_home)

     from app import app as application
     ```
   * Klik tombol hijau **Reload**. Web langsung aktif di `https://USERNAME.pythonanywhere.com`!

---

## Rangkuman Presentasi 10 Menit di Depan Bu Izzah

| Menit | Alur Presentasi | Poin yang Disampaikan |
| :--- | :--- | :--- |
| **00:00 - 02:00** | **Pendahuluan & Jurnal** | Kelompok kami memilih topik **Prediksi Data Time Series** dengan objek penjualan mobil Toyota di Indonesia. Rujukan utama kami adalah karya ilmiah dari kampus kita sendiri: **Jurnal Informatika Polinema (JIP) November 2024**. |
| **02:00 - 04:00** | **Dataset & Preprocessing** | Buka menu **Dataset GAIKINDO**. Jelaskan data bersumber dari GAIKINDO (149 bulan dari 2011 s.d. 2023). Tunjukkan preprocessing: normalisasi MinMaxScaler dan pembentukan fitur Lag-10 (menggunakan 10 bulan sebelumnya untuk memprediksi bulan berikutnya). Pembagian data latih 133 sampel dan data uji 6 bulan terakhir. |
| **04:00 - 06:30** | **Model & Evaluasi (Poin 7)** | Buka menu **Evaluasi & Komparasi**. Tunjukkan arsitektur MLP 10-15-1 dengan aktivasi ReLU dan optimizer Adam. Tunjukkan nilai MAE = 1.947 unit dan MSE, serta grafik garis Chart.js yang memperlihatkan kedekatan garis tebakan model dengan data riil GAIKINDO. Jelaskan bahwa MAPE 7.8% masuk kategori Sangat Akurat. |
| **06:30 - 08:30** | **Demo Prediksi Manual** | Buka menu **Uji Prediksi Manual**. Tunjukkan bahwa dosen bisa memasukkan angka penjualan 10 bulan terakhir sesuka hati atau klik contoh skenario (Normal, Krisis, Booming). Klik prediksi, dan tunjukkan tebakan bulan ke-11 beserta visualisasi grafiknya. |
| **08:30 - 10:00** | **Komparasi Jurnal (Poin 8)** | Tunjukkan bagian komparasi di bawah halaman Evaluasi. Jelaskan persamaan metode dan data GAIKINDO, perbedaan MAE (1.947 unit vs 1.879 unit di jurnal), serta analisis penyebabnya (perbedaan random seed inisialisasi bobot awal neural network pada Scikit-Learn dengan selisih yang sangat tipis). Tutup presentasi. |
