import pandas as pd
import faiss
import numpy as np

def initialize_faiss():
    # Load embedding dari file CSV
    df = pd.read_csv("./Belajar-DeepFace/face_embeddings.csv")

    # Convert embedding dari string ke list (jika disimpan sebagai string)
    df['embedding'] = df['embedding'].apply(eval)

    # Extract embedding dan konversi ke numpy array
    embeddings = np.array(df['embedding'].tolist(), dtype='f')

    # Inisialisasi FAISS
    num_dimensions = 128  # Dimensi embedding (untuk Facenet)
    index = faiss.IndexFlatL2(num_dimensions)

    # Tambahkan embedding ke indeks FAISS
    index.add(embeddings)

    # Simpan indeks FAISS jika diperlukan
    faiss.write_index(index, "./Belajar-Deepface/faiss_index.bin")

    print("FAISS index created and saved to faiss_index.bin")
