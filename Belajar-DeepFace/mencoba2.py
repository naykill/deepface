import streamlit as st
import pandas as pd
import os
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Folder untuk menyimpan file foto
FOTO_FOLDER = 'foto_user'
if not os.path.exists(FOTO_FOLDER):
    os.makedirs(FOTO_FOLDER)

# File CSV tempat menyimpan data
CSV_FILE = 'data_user.csv'

# Fungsi untuk menyimpan data ke CSV
def simpan_data(nama, posisi, foto_path):
    # Cek apakah file CSV sudah ada
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=["Nama", "Posisi", "Foto", "Tanggal"])
        df.to_csv(CSV_FILE, index=False)
    
    # Membaca data yang sudah ada
    df = pd.read_csv(CSV_FILE)

    # Menambahkan data baru
    tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({"Nama": [nama], "Posisi": [posisi], "Foto": [foto_path], "Tanggal": [tanggal]})
    
    # Menggunakan pd.concat untuk menambahkan data baru
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Menyimpan data kembali ke file CSV
    df.to_csv(CSV_FILE, index=False)

# Fungsi API untuk menerima data dari perangkat lain
@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        data = request.json
        nama = data.get("nama")
        posisi = data.get("posisi")
        foto_data = data.get("foto")  # Foto dalam bentuk string base64
        
        if not nama or not posisi or not foto_data:
            return jsonify({"status": "error", "message": "Nama, posisi, atau foto tidak lengkap."}), 400
        
        # Simpan foto
        foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        with open(foto_path, "wb") as f:
            f.write(foto_data.encode('utf-8'))  # Mengonversi string base64 ke binary
        
        # Simpan data ke CSV
        simpan_data(nama, posisi, foto_path)
        
        return jsonify({"status": "success", "message": "Data berhasil disimpan."})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Fungsi untuk antarmuka Streamlit
def streamlit_app():
    st.title("Form Pendaftaran dan Foto Pengguna")
    st.write("Silakan masukkan nama, posisi, dan ambil foto.")

    # Input form dari antarmuka web
    nama = st.text_input("Nama")
    posisi = st.text_input("Posisi")
    foto = st.camera_input("Ambil Foto")

    # Simpan data dan foto ketika tombol diklik
    if st.button("Simpan Data"):
        if nama and posisi and foto:
            foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            with open(foto_path, "wb") as f:
                f.write(foto.getbuffer())
            
            simpan_data(nama, posisi, foto_path)
            st.success(f"Data berhasil disimpan! Foto disimpan di: {foto_path}")
        else:
            st.error("Mohon lengkapi semua data (nama, posisi, dan foto).")

# Jalankan Flask untuk API
if __name__ == '__main__':
    from threading import Thread
    
    # Jalankan aplikasi Flask di thread terpisah agar bisa berjalan bersamaan dengan Streamlit
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    flask_thread.start()
    
    # Jalankan aplikasi Streamlit
    streamlit_app()
