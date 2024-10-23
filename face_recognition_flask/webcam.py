import cv2
import base64
import requests
import numpy as np
import time

# Inisialisasi webcam
cap = cv2.VideoCapture(1)  # Ganti dengan 1 jika menggunakan webcam eksternal
if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka. Pastikan kamera terhubung dan berfungsi.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SERVER_URL = "http://172.254.2.153:5000"

# Set interval pengambilan gambar (3 detik)
capture_interval = 3
start_time = time.time()

while True:
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
                        print(f"Selamat datang {data['name']} - {data['position']}")
                    else:
                        print(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    print(f"Error connecting to server: {e}")

            start_time = current_time

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