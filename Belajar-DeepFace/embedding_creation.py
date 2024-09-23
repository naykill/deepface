from deepface import DeepFace
import os
import pandas as pd

# Dictionary untuk mapping nama file ke nama orang
file_to_name = {
    "img5.jpg" : "Angelina Jolie",
    "img6.jpg" : "Angelina Jolie",
    "img14.jpg" : "Mark Zuckerberg",
    "img15.jpg" : "Mark Zuckerberg",
    "img17.jpg" : "Jack Dorsey",
    "img54.jpg" : "Jennifer Aniston",
    "img56.jpg" : "Jennifer Aniston",
    "img62.jpg" : "Jack Dorsey",
    "heydar.jpg" : "Heydar",
    "ica.jpg" : "Ica",
    "nola.jpg" : "Nola",
    "indah.jpg" : "Indah",
    "irsan.jpg" : "Irsan",
    "zeyad.jpg" : "Zeyad"
}
name_to_position = {
    "Angelina Jolie" : "Oscar Actress",
    "Mark Zuckerberg" : "CEO Meta",
    "Jack Dorsey" : "CEO Square",
    "Jennifer Aniston" : "Friends serial actress",
    "Heydar" : "walkot",
    "Ica" : "Engineer",
    "Nola" : "CEO rokan hulu",
    "Indah" : "Dirut elnusa",
    "Irsan" : "bupati sleman",
    "Zeyad" : "raja arab"
}

# Inisialisasi DataFrame Pandas untuk menyimpan nama dan embedding
df = pd.DataFrame(columns=["name", "embedding"])

model_name = "Facenet"  # Nama model DeepFace
detector_backend = "opencv"  # Backend untuk deteksi wajah

# Loop melalui folder dan buat embedding
for r, d, files in os.walk("../Belajar-DeepFace"):
    for file in files:
        if ".jpg" in file:
            exact_file = f"{r}/{file}"

            # Generate embedding wajah
            objs = DeepFace.represent(
                img_path=exact_file,
                model_name=model_name,
                detector_backend=detector_backend
            )
            
            # Dapatkan nama dari mapping
            name = file_to_name.get(file, "Unknown")
            posisi = name_to_position.get(name, "unknown position")
            # Simpan embedding ke dalam DataFrame
            for obj in objs:
                embedding = obj["embedding"]
                new_row = pd.DataFrame({"name": [name], "embedding": [embedding], "posisi": [posisi]})
                df = pd.concat([df, new_row], ignore_index=True)

# Simpan embedding dan nama ke dalam file CSV
df.to_csv("face_embeddings.csv", index=False)

print("Embeddings telah disimpan dengan nama file face_embeddings.csv")
