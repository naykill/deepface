from deepface import DeepFace
import cv2
import faiss
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def recognize_face_from_webcam():
    # Load indeks FAISS dan DataFrame embedding
    index = faiss.read_index("./Belajar-Deepface/faiss_index.bin")
    df = pd.read_csv("./Belajar-DeepFace/face_embeddings.csv")
    df['embedding'] = df['embedding'].apply(eval)

    # Inisialisasi webcam dan deteksi wajah
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Set interval pengambilan gambar (10 detik)
    capture_interval = 10
    start_time = time.time()

    while True:
        # Read frame dari webcam
        ret, frame = cap.read()

        # Detect wajah
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Jika wajah terdeteksi, capture wajah dan cari kecocokan terdekat
        if len(faces) > 0:
            current_time = time.time()

            if current_time - start_time >= capture_interval:
                for (x, y, w, h) in faces:
                    # Ekstrak wajah dari frame
                    face = frame[y:y+h, x:x+w]

                    # Konversi ke RGB
                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    # Generate embedding untuk wajah yang di-capture
                    target_embedding = DeepFace.represent(
                        img_path=rgb_face,
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=False
                    )[0]["embedding"]

                    # Convert embedding ke numpy dan expand dimensinya
                    target_embedding = np.expand_dims(np.array(target_embedding, dtype='f'), axis=0)

                    # Cari embedding terdekat di FAISS
                    k = 1
                    distances, neighbours = index.search(target_embedding, k)

                    # Get closest match
                    closest_match_idx = neighbours[0][0]

                    if closest_match_idx < len(df):
                        match_name = df.iloc[closest_match_idx]['name']
                        position = df.iloc[closest_match_idx]['posisi']
                        print(f"Match found: {match_name}")
                        print(f"Hallo {match_name} {position}, selamat datang")
                        # Tampilkan wajah dan match
                        fig = plt.figure(figsize=(7, 7))
                        fig.add_subplot(1, 2, 1)
                        plt.imshow(rgb_face)
                        plt.axis("off")
                        fig.add_subplot(1, 2, 2)
                        plt.imshow(plt.imread(f"./Belajar-DeepFace/{df.iloc[closest_match_idx]['name']}.jpg"))
                        plt.axis("off")
                        plt.show()

                # Reset time waktu capture dari webcam secara otomatis
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