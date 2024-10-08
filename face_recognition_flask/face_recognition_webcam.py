import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from deepface import DeepFace
import faiss
import pandas as pd
import time

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert frame ke format RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)

        # Jika ada wajah yang terdeteksi
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Tampilkan kotak hijau di sekitar wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Ekstrak wajah dari frame
                face = frame[y:y+h, x:x+w]
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Generate embedding dan cari match di FAISS
                try:
                    target_embedding = DeepFace.represent(img_path=rgb_face, model_name="Facenet", detector_backend="opencv")[0]["embedding"]
                    target_embedding = np.expand_dims(np.array(target_embedding, dtype='f'), axis=0)

                    # Cari embedding di FAISS index
                    k = 1
                    distances, neighbours = index.search(target_embedding, k)
                    closest_match_idx = neighbours[0][0]

                    if closest_match_idx < len(df):
                        match_name = df.iloc[closest_match_idx]['name']
                        position = df.iloc[closest_match_idx]['posisi']
                        
                        # Update informasi yang ditampilkan di GUI (gambar + nama + posisi)
                        update_info(match_name, position)

                except Exception as e:
                    print(f"Error: {e}")

        # Convert frame ke format Image untuk Tkinter
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
    lmain.after(10, update_frame)

def update_info(name, position):
    # Tampilkan nama dan posisi pada label
    name_label.config(text=f"Name: {name}")
    position_label.config(text=f"Position: {position}")

    # Load foto pengguna untuk ditampilkan
    try:
        img = Image.open(f"./Belajar-DeepFace/{name}.jpg")
        img = img.resize((100, 100))  # Resize gambar
        img_photo = ImageTk.PhotoImage(img)
        user_image_label.config(image=img_photo)
        user_image_label.image = img_photo
    except Exception as e:
        print(f"Error loading image: {e}")

# Setup FAISS dan DataFrame embedding
index = faiss.read_index("./Belajar-DeepFace/faiss_index.bin")
df = pd.read_csv("./Belajar-DeepFace/face_embeddings.csv")
df['embedding'] = df['embedding'].apply(eval)

# Inisialisasi webcam
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi Tkinter GUI
root = tk.Tk()
root.title("Attendance System")

# Webcam frame
lmain = tk.Label(root)
lmain.pack(side=tk.LEFT)

# Bagian untuk menampilkan informasi pengguna
info_frame = tk.Frame(root)
info_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Label untuk gambar pengguna
user_image_label = tk.Label(info_frame)
user_image_label.pack()

# Label untuk nama dan posisi
name_label = tk.Label(info_frame, text="Name: ", font=('Helvetica', 12))
name_label.pack()
position_label = tk.Label(info_frame, text="Position: ", font=('Helvetica', 12))
position_label.pack()

# Mulai update frame webcam dan informasi
update_frame()

# Start GUI main loop
root.mainloop()
