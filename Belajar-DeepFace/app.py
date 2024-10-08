from flask import Flask, request, jsonify
import os
import pandas as pd
from embedding_creation import create_embedding
import json
from datetime import datetime

app = Flask(__name__)

# Folder untuk menyimpan data
CSV_FILE = 'face_embeddings.csv'
JSON_FILE = 'face_embeddings.json'

# Fungsi untuk menyimpan data ke CSV dan JSON
def save_to_csv_json(nama, posisi, embedding):
    # Simpan ke CSV
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Nama", "Posisi", "Embedding"])
    else:
        df = pd.read_csv(CSV_FILE)

    new_data = {
        "Nama": nama,
        "Posisi": posisi,
        "Embedding": embedding
    }
    df = df.append(new_data, ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    # Simpan ke JSON
    data_json = df.to_dict(orient="records")
    with open(JSON_FILE, "w") as json_file:
        json.dump(data_json, json_file, indent=4)

# Route untuk pendaftaran pengguna
@app.route('/register', methods=['POST'])
def register():
    nama = request.form['nama']
    posisi = request.form['posisi']
    foto = request.files['foto']

    # Simpan foto sementara
    foto_path = f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    foto.save(foto_path)

    # Buat embedding dari foto
    embedding = create_embedding(foto_path)

    # Simpan ke CSV dan JSON
    save_to_csv_json(nama, posisi, embedding)

    return jsonify({"message": "Data berhasil disimpan"}), 200

# Route untuk memberikan data JSON (digunakan oleh face recognition)
@app.route('/get-json', methods=['GET'])
def get_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as json_file:
            data = json.load(json_file)
        return jsonify(data), 200
    return jsonify({"error": "File JSON tidak ditemukan"}), 404

if __name__ == '__main__':
    app.run(debug=True)
