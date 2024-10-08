import streamlit as st  # Digunakan untuk membangun antarmuka aplikasi web interaktif
import pandas as pd  # Untuk mengelola data tabular dan menyimpan/menambah data ke file CSV
import os  # Untuk berinteraksi dengan sistem file
from PIL import Image  # Library untuk memanipulasi gambar
import cv2  # Library untuk deteksi dan manipulasi gambar

# Folder untuk menyimpan foto pengguna
FOTO_FOLDER = 'foto_user'
if not os.path.exists(FOTO_FOLDER):
    os.makedirs(FOTO_FOLDER)

# Folder untuk menyimpan hasil crop wajah
WAJAH_FOLDER = 'foto_wajah'
if not os.path.exists(WAJAH_FOLDER):
    os.makedirs(WAJAH_FOLDER)

# File CSV tempat menyimpan data
CSV_FILE = 'data_user.csv'

# Membuat form input untuk nama dan posisi
st.title("Form Pendaftaran dan Foto Pengguna")
st.write("Silakan masukkan nama, posisi, dan ambil foto.")

# Mengambil input nama dan posisi
nama = st.text_input("Nama")
posisi = st.text_input("Posisi")

# Fungsi untuk menyimpan data ke CSV
def simpan_data(nama, posisi, foto_path):
    # Cek apakah file CSV sudah ada
    if not os.path.isfile(CSV_FILE):
        # Buat file CSV dengan header jika belum ada
        df = pd.DataFrame(columns=["Nama", "Posisi", "Foto"])
        df.to_csv(CSV_FILE, index=False)
    
    # Membaca data yang sudah ada
    df = pd.read_csv(CSV_FILE)

    # Menambah data baru
    new_data = pd.DataFrame({"Nama": [nama], "Posisi": [posisi], "Foto": [foto_path]})
    
    # Menggunakan pd.concat untuk menambahkan data baru
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Menyimpan data kembali ke file CSV
    df.to_csv(CSV_FILE, index=False)

# Fungsi untuk mendeteksi dan memotong wajah menggunakan OpenCV
def deteksi_dan_crop_wajah(image_path):
    # Load gambar
    img = cv2.imread(image_path)
    
    # Load classifier untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Konversi gambar ke skala abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Jika wajah terdeteksi, potong wajah
    for (x, y, w, h) in faces:
        wajah = img[y:y + h, x:x + w]
        # Simpan hasil crop ke folder berbeda
        wajah_path = os.path.join(WAJAH_FOLDER, os.path.basename(image_path).replace(".png", "_wajah.png"))
        cv2.imwrite(wajah_path, wajah)
        return wajah_path
    
    # Jika tidak ada wajah yang terdeteksi, return None
    return None

# Mengambil foto dari input kamera
foto = st.camera_input("Ambil Foto")

# Simpan data dan foto ketika tombol diklik
if st.button("Simpan Data"):
    if nama and posisi and foto:
        # Simpan foto asli tanpa timestamp
        foto_path = os.path.join(FOTO_FOLDER, f"{nama}.png")
        with open(foto_path, "wb") as f:
            f.write(foto.getbuffer())
        
        # Crop wajah dari foto yang diambil
        wajah_path = deteksi_dan_crop_wajah(foto_path)
        
        if wajah_path:
            # Simpan data ke CSV dengan path wajah yang dipotong
            simpan_data(nama, posisi, wajah_path)
            st.success(f"Data berhasil disimpan! Wajah disimpan di: {wajah_path}")
        else:
            st.warning("Wajah tidak terdeteksi pada foto. Data disimpan dengan foto asli.")
            simpan_data(nama, posisi, foto_path)
        
    else:
        st.error("Mohon lengkapi semua data (nama, posisi, dan foto).")