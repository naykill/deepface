import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# Judul aplikasi
st.title("Pendaftaran Karyawan dengan Foto")

# Input nama dan posisi
name = st.text_input("Masukkan Nama Karyawan")
position = st.text_input("Masukkan Posisi Karyawan")

# Upload foto
uploaded_file = st.file_uploader("Unggah Foto Karyawan", type=["jpg", "jpeg", "png"])

# Tombol untuk mengirim data
if st.button("Daftarkan Karyawan"):
    if uploaded_file is not None and name and position:
        try:
            # Baca file gambar dan konversi ke numpy array
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # Konversi gambar ke format byte untuk dikirim melalui API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()

            # Kirim data ke API Flask untuk membuat embedding
            url = "http://127.0.0.1:5000/create-embeddings"
            files = {"file": img_bytes}
            data = {"name": name, "position": position}

            # Kirim request POST ke API Flask
            response = requests.post(url, files={"file": uploaded_file}, data=data)

            # Tampilkan hasil
            if response.status_code == 200:
                st.success(f"Karyawan {name} berhasil didaftarkan!")
            else:
                st.error(f"Terjadi kesalahan: {response.json().get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.warning("Mohon lengkapi nama, posisi, dan foto karyawan.")