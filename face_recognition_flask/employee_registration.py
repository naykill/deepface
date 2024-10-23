import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

# Function to convert file image to base64
def convert_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

SERVER_URL = "http://172.254.2.153:5000"

# List of positions available
positions = [
    "President Director", "IT Manager", "Software Engineer", "Data Analyst", 
    "IT Support", "Quality Assurance", "Product Manager", "HR Manager", 
    "Accountant", "Marketing Specialist", "UI/UX Designer", "Business Analyst"
]

# Sidebar with menu options
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Karyawan", "Data Karyawan", "Edit/Hapus Karyawan","Log Absensi"])

# Function to crop face from the image
def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]  # Return the cropped face
    return None

# Function to read image as numpy array from Streamlit file input
def read_image(uploaded_image):
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, 1)  # Decode the image as OpenCV format
    return image

# Employee registration
if menu == "Pendaftaran Karyawan":
    st.title("Pendaftaran Karyawan")

    name = st.text_input("Nama Karyawan")
    position = st.selectbox("Posisi Karyawan", positions)
    foto = st.camera_input("Capture Foto Karyawan")
    uploaded_image = st.file_uploader("Unggah Foto Karyawan", type=["jpg", "jpeg", "png"])

    if st.button("Daftarkan Karyawan"):
        if foto and uploaded_image:
            st.error("Pilih salah satu: Capture foto atau unggah foto. Tidak boleh keduanya.")
        elif name and position and (foto or uploaded_image):
            if foto:
                image = read_image(foto)
            else:
                image = read_image(uploaded_image)

            cropped_face = crop_face(image)

            if cropped_face is not None:
                _, buffer = cv2.imencode('.jpg', cropped_face)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                data = {"name": name, "position": position, "image": image_base64}
                response = requests.post(f"{SERVER_URL}/register-employee", json=data)

                if response.status_code == 200:
                    st.success("Karyawan berhasil didaftarkan!")
                else:
                    st.error(f"Error: {response.text}")
            else:
                st.error("Wajah tidak terdeteksi di gambar, coba lagi.")
        else:
            st.warning("Mohon lengkapi semua data sebelum mengirim.")

# List employees with photos
elif menu == "Data Karyawan":
    st.title("Data Karyawan")

    response = requests.get(f"{SERVER_URL}/employees-info")

    if response.status_code == 200:
        employees = response.json()
        if employees:
            # Create a list to hold the rows of our table
            table_data = []
            
            for emp in employees:
                # Convert base64 image to displayable format
                image_data = base64.b64decode(emp['image_base64'])
                img = Image.open(BytesIO(image_data))
                
                # Resize image to make it smaller in the table
                img.thumbnail((100, 100))
                
                # Convert PIL Image to bytes
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create HTML for the image
                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100px;">'
                
                # Add this employee's data to our table
                table_data.append([emp['name'], emp['position'], img_html])
            
            # Create DataFrame
            df = pd.DataFrame(table_data, columns=["Nama", "Posisi", "Foto"])
            
            # Display the table
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("Tidak ada data karyawan yang tersedia.")
    else:
        st.error(f"Gagal mengambil data karyawan. Error: {response.text}")

# Edit or Delete employees
elif menu == "Edit/Hapus Karyawan":
    st.title("Edit atau Hapus Data Karyawan")

    response = requests.get("http://127.0.0.1:5000/employees-full")
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            employee = st.selectbox("Pilih Karyawan untuk Diedit atau Dihapus", df["name"])
            selected_employee = df[df["name"] == employee].iloc[0]

            st.subheader("Edit Data Karyawan")
            updated_name = st.text_input("Nama Karyawan", value=selected_employee["name"])
            updated_position = st.selectbox("Posisi Karyawan", positions, index=positions.index(selected_employee["position"]))

            if st.button("Simpan Perubahan"):
                update_data = {"name": updated_name, "position": updated_position}
                response = requests.put(f"{SERVER_URL}/update-employee/{selected_employee['id']}", json=update_data)

                if response.status_code == 200:
                    st.success("Data karyawan berhasil diperbarui!")
                else:
                    st.error(f"Error: {response.text}")

            st.subheader("Hapus Karyawan")
            if st.button("Hapus Karyawan"):
                response = requests.delete(f"{SERVER_URL}/delete-employee/{selected_employee['id']}")
                if response.status_code == 200:
                    st.success("Karyawan berhasil dihapus!")
                else:
                    st.error(f"Error: {response.text}")

#menu log
elif menu == "Log Absensi":
    st.title("Log Absensi Karyawan")
    
    # Tambahkan filter
    col1, col2 = st.columns(2)
    with col1:
        # Get all employees for filter
        emp_response = requests.get(f"{SERVER_URL}/employees-full")
        if emp_response.status_code == 200:
            employees = emp_response.json()
            employee_names = ["Semua Karyawan"] + [emp["name"] for emp in employees]
            selected_employee = st.selectbox("Filter berdasarkan Karyawan:", employee_names)
    
    with col2:
        filter_date = st.date_input("Filter berdasarkan Tanggal:")
    
    # Tombol untuk melihat log
    if st.button("Tampilkan Log Absensi"):
        if selected_employee == "Semua Karyawan":
            response = requests.get("http://127.0.0.1:5000/attendance-records")
        else:
            response = requests.get(f"http://127.0.0.1:5000/attendance-records/{selected_employee}")
        
        if response.status_code == 200:
            attendance_data = response.json()
            
            if attendance_data:
                # Filter berdasarkan tanggal jika dipilih
                if filter_date:
                    attendance_data = [
                        record for record in attendance_data 
                        if record['date'] == filter_date.strftime('%Y-%m-%d')
                    ]
                
                if attendance_data:
                    # Convert data to DataFrame
                    df = pd.DataFrame(attendance_data)
                    
                    # Tambahkan kolom untuk foto
                    if 'image_capture' in df.columns:
                        # Fungsi untuk mengkonversi base64 ke HTML img
                        def create_image_html(base64_str):
                            if base64_str:
                                return f'<img src="data:image/jpeg;base64,{base64_str}" style="width:100px;">'
                            return ""
                        
                        df['Foto Absensi'] = df['image_capture'].apply(create_image_html)
                        df = df.drop(columns=['image_capture'])
                    
                    # Rename columns for better display
                    df = df.rename(columns={
                        'employee_name': 'Nama Karyawan',
                        'date': 'Tanggal',
                        'time': 'Jam',
                        'status': 'Status'
                    })
                    
                    # Format tanggal dan waktu
                    df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.strftime('%d-%m-%Y')
                    
                    # Sort by date and time
                    df = df.sort_values(['Tanggal', 'Jam'], ascending=[False, False])
                    
                    # Display the table
                    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Tambahkan statistik
                    st.subheader("Statistik Absensi")
                    if selected_employee == "Semua Karyawan":
                        # Statistik per karyawan
                        st.write("Jumlah Absensi per Karyawan:")
                        attendance_count = df['Nama Karyawan'].value_counts()
                        st.bar_chart(attendance_count)
                    
                    # Statistik per tanggal
                    st.write("Jumlah Absensi per Tanggal:")
                    date_count = df['Tanggal'].value_counts()
                    st.line_chart(date_count)
                    
                else:
                    st.info("Tidak ada data absensi untuk filter yang dipilih.")
            else:
                st.info("Belum ada data absensi.")
        else:
            st.error(f"Gagal mengambil data absensi: {response.text}")

    # Tambahkan opsi ekspor data
    if st.button("Ekspor Data ke CSV"):
        if 'df' in locals():  # Check if DataFrame exists
            # Convert DataFrame to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            
            # Download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'absensi_{filter_date.strftime("%Y-%m-%d") if filter_date else "all"}.csv',
                mime='text/csv',
            )