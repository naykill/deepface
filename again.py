import streamlit as st  # Digunakan untuk membangun antarmuka aplikasi web interaktif.
import pandas as pd  # Untuk mengelola data tabular dan menyimpan/menambah data ke file CSV.
import os  # Untuk berinteraksi dengan sistem file, seperti membuat folder untuk menyimpan foto.
from PIL import Image  # Library untuk memanipulasi gambar.
import cv2  # Library OpenCV untuk deteksi wajah.
from datetime import datetime  # Untuk mendapatkan timestamp (tanggal dan waktu saat ini) untuk menyimpan data.

FOTO_FOLDER = 'foto_user'  # Folder untuk menyimpan file foto.
if not os.path.exists(FOTO_FOLDER):  # Membuat folder jika belum ada.
    os.makedirs(FOTO_FOLDER)

CSV_FILE = 'data_user.csv'  # File CSV tempat menyimpan data.

# Membuat form input untuk nama dan posisi.
st.title("Form Pendaftaran dan Foto Pengguna")
st.write("Silakan masukkan nama, posisi, dan ambil foto.")

# Mengambil input nama dan posisi.
nama = st.text_input("Nama")
posisi = st.text_input("Posisi")

# Fungsi untuk menyimpan data ke CSV.
def simpan_data(nama, posisi, foto_path):
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=["Nama", "Posisi", "Foto", "Tanggal"])
        df.to_csv(CSV_FILE, index=False)

    df = pd.read_csv(CSV_FILE)

    # Menambah data baru
    tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({"Nama": [nama], "Posisi": [posisi], "Foto": [foto_path], "Tanggal": [tanggal]})
    
    df = pd.concat([df, new_data], ignore_index=True)
    
    df.to_csv(CSV_FILE, index=False)

# Fungsi untuk mendeteksi dan crop wajah
def deteksi_wajah(foto_path):
    # Load gambar menggunakan OpenCV
    img = cv2.imread(foto_path)

    # Mengubah gambar menjadi grayscale untuk deteksi wajah
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load model deteksi wajah OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop wajah
            face = img[y:y+h, x:x+w]
            # Simpan gambar hasil crop
            cropped_face_path = os.path.join(FOTO_FOLDER, f"{nama}_cropped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(cropped_face_path, face)
            st.image(cropped_face_path, caption="Hasil Crop Wajah")
            return cropped_face_path
    else:
        st.warning("Wajah tidak terdeteksi dalam gambar.")
        return None

# Mengambil foto.
foto = st.camera_input("Ambil Foto")

# Simpan data dan foto ketika tombol diklik.
if st.button("Simpan Data"):
    if nama and posisi and foto:
        # Simpan foto.
        foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        with open(foto_path, "wb") as f:
            f.write(foto.getbuffer())

        # Deteksi dan crop wajah
        cropped_face_path = deteksi_wajah(foto_path)

        if cropped_face_path:
            # Simpan data ke CSV jika wajah berhasil terdeteksi dan di-crop
            simpan_data(nama, posisi, cropped_face_path)
            st.success(f"Data berhasil disimpan! Foto wajah disimpan di: {cropped_face_path}")
        else:
            st.error("Gagal mendeteksi wajah. Data tidak disimpan.")
    else:
        st.error("Mohon lengkapi semua data (nama, posisi, dan foto).")
