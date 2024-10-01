import streamlit as st #Digunakan untuk membangun antarmuka aplikasi web interaktif.
import pandas as pd #Untuk mengelola data tabular dan menyimpan/menambah data ke file CSV.
import os #Untuk berinteraksi dengan sistem file, seperti membuat folder untuk menyimpan foto.
from PIL import Image #Library untuk memanipulasi gambar. Namun, di sini hanya digunakan jika perlu (untuk pengembangan lebih lanjut).
from datetime import datetime #untuk mendapatkan timestamp (tanggal dan waktu saat ini) untuk menyimpan data.



FOTO_FOLDER = 'foto_user' #folder untuk menyimpan file foto
if not os.path.exists(FOTO_FOLDER): #membuat folder jika belum ada
    os.makedirs(FOTO_FOLDER)

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
        df = pd.DataFrame(columns=["Nama", "Posisi", "Foto", "Tanggal"])
        df.to_csv(CSV_FILE, index=False)
    
    # Membaca data yang sudah ada
    df = pd.read_csv(CSV_FILE)
    
    # Menambah data baru
    tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = df.append({"Nama": nama, "Posisi": posisi, "Foto": foto_path, "Tanggal": tanggal}, ignore_index=True)
    
    # Menyimpan data kembali ke file CSV
    df.to_csv(CSV_FILE, index=False)

# Mengambil foto
foto = st.camera_input("Ambil Foto")

# Simpan data dan foto ketika tombol diklik
if st.button("Simpan Data"):
    if nama and posisi and foto:
        # Simpan foto
        foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        with open(foto_path, "wb") as f:
            f.write(foto.getbuffer())
        
        # Simpan data ke CSV
        simpan_data(nama, posisi, foto_path)
        
        st.success(f"Data berhasil disimpan! Foto disimpan di: {foto_path}")
    else:
        st.error("Mohon lengkapi semua data (nama, posisi, dan foto).")




st.write(1234)
st.write(
    pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
        }
    )
)