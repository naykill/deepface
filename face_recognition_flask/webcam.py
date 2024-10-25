import cv2
import base64
import requests
import numpy as np
import time
from datetime import datetime

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # Ganti dengan 1 jika menggunakan webcam eksternal
if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka. Pastikan kamera terhubung dan berfungsi.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SERVER_URL = "http://172.254.2.153:5000"

# ESP32 URLs for controlling the gate
ESP32_URL_OPEN = "http://172.254.2.78/open-gate"   # Replace with your ESP32 IP
ESP32_URL_CLOSE = "http://172.254.2.78/close-gate" # Replace with your ESP32 IP

# Set interval pengambilan gambar (3 detik)
capture_interval = 3
start_time = time.time()

# Tambahkan variabel untuk tracking absensi
attendance_recorded = set()  # Untuk mencatat siapa saja yang sudah absen hari ini
current_date = datetime.now().strftime('%Y-%m-%d')

def reset_attendance_record():
    global attendance_recorded, current_date
    new_date = datetime.now().strftime('%Y-%m-%d')
    if new_date != current_date:
        attendance_recorded.clear()
        current_date = new_date

while True:
    reset_attendance_record()  # Reset data absensi jika hari berganti
    
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca dari webcam.")
        break

    # Deteksi wajah
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Jika wajah terdeteksi, capture wajah
    if len(faces) > 0:
        current_time = time.time()
        if current_time - start_time >= capture_interval:
            for (x, y, w, h) in faces:
                # Ekstrak wajah dari frame
                face = frame[y:y+h, x:x+w]

                # Konversi wajah ke base64
                _, buffer = cv2.imencode('.jpg', face)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Kirim gambar ke server untuk identifikasi
                try:
                    response = requests.post(f"{SERVER_URL}/identify-employee", 
                                             json={"image": image_base64},
                                             headers={'Content-Type': 'application/json'})
                    if response.status_code == 200:
                        data = response.json()
                        employee_name = data['name']
                        # Cek apakah karyawan sudah absen hari ini
                        if employee_name not in attendance_recorded:
                            # Catat absensi
                            attendance_response = requests.post(
                                f"{SERVER_URL}/record-attendance",
                                json={
                                    "name": employee_name,
                                    "image": image_base64,
                                    "status": "masuk"
                                }
                            )
                            
                            if attendance_response.status_code == 200:
                                attendance_data = attendance_response.json()
                                print(f"Absensi berhasil: {employee_name}")
                                print(f"Tanggal: {attendance_data['date']}")
                                print(f"Jam: {attendance_data['time']}")
                                attendance_recorded.add(employee_name)
                            else:
                                print(f"Gagal mencatat absensi: {attendance_response.json()['message']}")
                        
                        print(f"Selamat datang {data['name']} - {data['position']}")

                        # Send request to ESP32 to open the gate
                        try:
                            esp_response = requests.get(ESP32_URL_OPEN)
                            if esp_response.status_code == 200:
                                print("Gate opened successfully!")
                        except requests.exceptions.RequestException as e:
                            print(f"Error connecting to ESP32: {e}")
                    else:
                        print(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    print(f"Error connecting to server: {e}")

            start_time = current_time

    else:
        # If no face is detected, send request to close the gate
        try:
            esp_response = requests.get(ESP32_URL_CLOSE)
            if esp_response.status_code == 200:
                print("Gate closed successfully!")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ESP32: {e}")

    # Tampilkan frame webcam dengan kotak di sekitar wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
