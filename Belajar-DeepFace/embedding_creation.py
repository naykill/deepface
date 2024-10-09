from deepface import DeepFace
import os
import pandas as pd

def create_embedding(image_folder):
    # Dictionary untuk mapping nama file ke nama orang
    file_to_name = {
        "Angelina Jolie.jpg" : "Angelina Jolie",
        "img6.jpg" : "Angelina Jolie",
        "Mark Zuckerberg.jpg" : "Mark Zuckerberg",
        "img15.jpg" : "Mark Zuckerberg",
        "Jack Dorsey.jpg" : "Jack Dorsey",
        "Jennifer Aniston.jpg" : "Jennifer Aniston",
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
    df = pd.DataFrame(columns=["name", "embedding", "posisi"])

    model_name = "Facenet"  # Nama model DeepFace
    detector_backend = "opencv"  # Backend untuk deteksi wajah

    # Loop melalui folder dan buat embedding
    for r, d, files in os.walk("./Belajar-DeepFace/"):
        for file in files:
            if ".jpg" in file:
                exact_file = f"{r}/{file}"
                print(f"Processing file: {exact_file}")

                try:
                    # Generate embedding wajah
                    objs = DeepFace.represent(
                        img_path=exact_file,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=False  # Set to False to handle images without clear faces
                    )
                    
                    if len(objs) > 0:
                        # Dapatkan nama dari mapping
                        name = file_to_name.get(file, "Unknown")  # Jika file tidak ada dalam dictionary, beri nama 'Unknown'
                        posisi = name_to_position.get(name, "unknown position")
                        
                        for obj in objs:
                            embedding = obj["embedding"]
                            # Buat DataFrame baru dari nama dan embedding yang dihasilkan
                            new_row = pd.DataFrame({"name": [name], "embedding": [embedding], "posisi": [posisi]})
                            
                            # Gabungkan DataFrame baru ke DataFrame utama menggunakan pd.concat()
                            df = pd.concat([df, new_row], ignore_index=True)
                    else:
                        print(f"No embeddings found for {file}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # Simpan embedding dan nama ke dalam file CSV
    if not df.empty:
        df.to_csv("./Belajar-DeepFace/face_embeddings.csv", index=False)
        print("Embeddings telah disimpan dengan nama file embeddings.csv")
    else:
        print("Tidak ada embedding yang disimpan. Pastikan file gambar memiliki wajah.")