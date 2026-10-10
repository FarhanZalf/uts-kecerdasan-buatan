# 🌸 Klasifikasi Spesies Bunga Iris Berbasis Flask & Decision Tree
> **Project Ujian Tengah Semester (UTS) — Mata Kuliah Kecerdasan Buatan**  
> **Program Studi D3 Manajemen Informatika — PSDKU Kota Kediri — Politeknik Negeri Malang**  
> **Dosen Pengampu:** Abidatul Izzah, M.Kom  

---

## 📌 Ringkasan Project
Aplikasi web berbasis **Flask** yang menerapkan algoritma Machine Learning **Decision Tree (CART dengan Gini Index)** serta pembanding **Multilayer Perceptron (MLP)** untuk mengklasifikasikan 3 spesies bunga Iris (*Setosa, Versicolor, Virginica*) berdasarkan 4 atribut numerik (*Sepal Length, Sepal Width, Petal Length, Petal Width*).

Aplikasi ini memenuhi seluruh kriteria soal UTS:
1. ✅ **Sumber & Karakteristik Dataset:** Bersumber dari UCI Machine Learning Repository (150 sampel, 4 fitur, no missing values).
2. ✅ **Preprocessing & Data Split:** Pembersihan data, normalisasi, dan pembagian data latih (80%) dan data uji (20%) dengan *stratified sampling* serta *10-Fold Cross-Validation*.
3. ✅ **Model Klasifikasi:** Decision Tree (CART Gini) dan Multilayer Perceptron (MLP).
4. ✅ **Evaluasi Model:** Confusion Matrix, Akurasi (93.33%), Precision (93.33%), Recall (93.33%), F1-Score (93.33%).
5. ✅ **Pengujian Data Baru:** Form interaktif live testing di web dengan probabilitas per kelas dan visualisasi foto bunga.
6. ✅ **Komparasi Artikel Jurnal (Poin 8 UTS):** Perbandingan mendalam dengan jurnal ilmiah referensi:  
   *Muhammad Dicky Azhary Octavianto & Tata Sutabri (2025). "Analisis Perbandingan Kinerja Algoritma Klasifikasi Data Menggunakan Metode K-NN, Naive Bayes, dan Decision Tree pada Dataset UCI Iris"*, Jurnal MEANS Vol. 10 No. 1.

---

## 📁 Struktur Direktori
```text
uts kb/
│
├── dataset/
│   └── iris.csv                 # Dataset UCI Iris (150 baris x 5 kolom)
│
├── models/
│   ├── decision_tree_model.pkl  # Model Decision Tree terlatih
│   ├── mlp_model.pkl            # Model Multilayer Perceptron terlatih
│   └── metrics.json             # Metadata metrik evaluasi & komparasi
│
├── static/
│   ├── css/
│   │   └── style.css            # Tampilan antarmuka modern & responsif
│   ├── js/
│   │   └── main.js              # Interaktivitas slider & quick presets
│   └── img/
│       ├── setosa.jpg           # Foto bunga Iris Setosa
│       ├── versicolor.jpg       # Foto bunga Iris Versicolor
│       └── virginica.jpg        # Foto bunga Iris Virginica
│
├── templates/
│   ├── base.html                # Layout dasar navbar & footer
│   ├── index.html               # Beranda & ringkasan arsitektur
│   ├── dataset.html             # Eksplorasi dataset & tabel interaktif
│   ├── evaluasi.html            # Confusion Matrix & metrik evaluasi
│   ├── prediksi.html            # Form live test pengujian data baru
│   └── komparasi.html           # Analisis komparasi jurnal (Poin 8)
│
├── app.py                       # Aplikasi web backend Flask
├── train_model.py               # Script training & kalkulasi metrik
├── wsgi.py                      # Konfigurasi deploy PythonAnywhere
├── requirements.txt             # Dependensi Python
├── .gitignore                   # Ignore cache/temporary files
└── README.md                    # Dokumentasi lengkap project
```

---

## 💻 Cara Menjalankan Secara Lokal

1. **Buka Terminal / Command Prompt** pada folder project ini:
   ```bash
   cd "e:\Semester 5\Kecerdasan Buatan\uts kb"
   ```

2. **Install Dependensi (jika belum terpasang):**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Opsional) Latih Ulang Model:**
   ```bash
   python train_model.py
   ```

4. **Jalankan Server Flask:**
   ```bash
   python app.py
   ```

5. **Buka di Browser:**
   Akses `http://127.0.0.1:5000` pada browser Anda.

---

## 🚀 Panduan Deploy ke PythonAnywhere (Via GitHub)

### Langkah 1: Push Project ke Akun GitHub Anda
1. Inisialisasi Git di folder ini:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - UTS Kecerdasan Buatan Flask Iris Classifier"
   ```
2. Buat repositori baru di GitHub (misal bernama `uts-kb-iris`), lalu hubungkan dan push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/USERNAME-ANDA/uts-kb-iris.git
   git push -u origin main
   ```

---

### Langkah 2: Deploy di PythonAnywhere
1. Buka [https://www.pythonanywhere.com](https://www.pythonanywhere.com) dan login ke akun gratis Anda.
2. Buka tab **Consoles** -> Klik **Bash**.
3. Di dalam terminal Bash PythonAnywhere, clone repositori GitHub Anda dan install dependensi:
   ```bash
   git clone https://github.com/USERNAME-ANDA/uts-kb-iris.git
   cd uts-kb-iris
   pip install -r requirements.txt
   ```
4. Buka tab **Web** di menu atas PythonAnywhere:
   * Klik tombol **Add a new web app**.
   * Pilih **Manual configuration**, lalu pilih **Python 3.10**.
5. Pada bagian **Code**:
   * Set **Source code**: `/home/USERNAME-ANDA/uts-kb-iris`
   * Set **Working directory**: `/home/USERNAME-ANDA/uts-kb-iris`
6. Pada bagian **WSGI configuration file**:
   * Klik link file WSGI (contoh: `/var/www/USERNAME_pythonanywhere_com_wsgi.py`).
   * Hapus seluruh isinya, lalu ganti dengan baris berikut:
     ```python
     import sys
     import os

     project_home = '/home/USERNAME-ANDA/uts-kb-iris'
     if project_home not in sys.path:
         sys.path.insert(0, project_home)

     from app import app as application
     ```
   * Klik tombol **Save** di pojok kanan atas.
7. Kembali ke tab **Web**, klik tombol hijau besar: **Reload USERNAME.pythonanywhere.com**.
8. **Selesai!** Web Anda langsung live di `https://USERNAME.pythonanywhere.com` dan dapat diakses oleh dosen secara online!

---

## 🎤 Contekan Presentasi 10 Menit di Depan Bu Izzah

| Menit | Alur Presentasi | Poin yang Disampaikan |
| :--- | :--- | :--- |
| **00:00 - 02:00** | **Pendahuluan & Jurnal** | Perkenalkan diri, sebutkan topik **Klasifikasi Data**, serta artikel jurnal acuan dari Octavianto & Sutabri (Jurnal MEANS 2025) yang meneliti dataset UCI Iris. |
| **02:00 - 04:00** | **Dataset & Preprocessing** | Tunjukkan menu **Dataset**, jelaskan 150 sampel, 4 fitur, pembagian 80:20 (120 data latih, 30 data uji) dengan stratified sampling. |
| **04:00 - 06:30** | **Model & Evaluasi** | Tunjukkan menu **Evaluasi**, jelaskan algoritma **Decision Tree (CART Gini)**, tunjukkan tabel **Confusion Matrix**, jelaskan akurasi 93.33%, presisi 93.33%, recall 93.33%, dan visualisasi pohon logika (*Tree Rules*). |
| **06:30 - 08:30** | **Demo Live Prediksi** | Tunjukkan menu **Uji Prediksi**, masukkan angka (atau klik contoh cepat), tunjukkan sistem memprediksi spesies bunga secara langsung disertai foto dan persentase keyakinan. |
| **08:30 - 10:00** | **Komparasi Jurnal (Poin 8)** | Tunjukkan menu **Komparasi Jurnal**, jelaskan persamaan metode CART Gini, perbedaan akurasi (93.33% vs 95.33%), dan analisis penyebab perbedaan (faktor partisi split vs 10-fold CV pada 150 data). |
