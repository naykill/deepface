import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
import sqlite3
from deepface import DeepFace
import numpy as np
import faiss
import base64
import cv2
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to SQLite database
db_path = './face_embeddings.db'

# Initialize SQLite database and create tables if they don't exist
def init_db():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Table for storing embeddings (already exists)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                position TEXT,
                embedding BLOB
            )
        ''')
        
        #new table for attendance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT NOT NULL,
                date DATE NOT NULL,
                jam_masuk TIME,
                jam_keluar TIME DEFAULT NULL,
                jam_kerja TEXT DEFAULT NULL,
                image_capture TEXT,
                status TEXT NOT NULL
            )
        ''')
        conn.commit()

        # New table for storing employee images and info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee_info (
                employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                position TEXT,
                image_base64 TEXT
            )
        ''')
        conn.commit()

# Insert employee data into both employee_info and employees tables
def insert_employee(name, position, embedding, image_base64):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Insert into employees (embedding table)
        cursor.execute("INSERT INTO employees (name, position, embedding) VALUES (?, ?, ?)", 
                       (name, position, embedding))

        # Insert into employee_info (image table)
        cursor.execute("INSERT INTO employee_info (name, position, image_base64) VALUES (?, ?, ?)", 
                       (name, position, image_base64))
        
        conn.commit()

def fetch_embeddings():
       with sqlite3.connect(db_path) as conn:
           cursor = conn.cursor()
           cursor.execute("SELECT id, name, position, embedding FROM employees")
           data = cursor.fetchall()

           encoded_data = []
           for row in data:
               id, name, position, embedding_blob = row
               embedding_base64 = base64.b64encode(embedding_blob).decode('utf-8')
               encoded_data.append({
                   "id": id,
                   "name": name,
                   "position": position,
                   "embedding": embedding_base64
               })
       
       return encoded_data

# Convert embeddings stored as blob back to numpy array
def convert_embedding(blob):
    return np.frombuffer(blob, dtype='f')

@app.route('/register-employee', methods=['POST'])
def register_employee():
    data = request.json
    name = data['name']
    position = data['position']
    image_base64 = data['image']  # The base64 image data

    model_name = "Facenet"
    detector_backend = "opencv"

    try:
        # Decode base64 image to process it with DeepFace
        image_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Generate embedding
        objs = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if len(objs) > 0:
            embedding = np.array(objs[0]['embedding'], dtype='f')
            embedding_blob = embedding.tobytes()  # Convert numpy array to binary

            # Insert employee into both tables (embedding and image data)
            insert_employee(name, position, embedding_blob, image_base64)

            return jsonify({"message": "Karyawan berhasil didaftarkan!"}), 200
        else:
            return jsonify({"message": "Tidak ada wajah terdeteksi."}), 400

    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/identify-employee', methods=['POST'])
def identify_employee():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"message": "No image data provided"}), 400

    image_base64 = data['image']
    model_name = "Facenet"
    detector_backend = "opencv"

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"message": "Failed to decode image"}), 400

        # Generate embedding
        target_embedding = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )[0]["embedding"]

        target_embedding = np.array(target_embedding, dtype='f')

        # Fetch all embeddings from the database
        employees_data = fetch_embeddings()

        # Extract embeddings and store them in a numpy array
        embeddings = []
        employee_details = []
        for emp in employees_data:
            name, position, embedding_blob = emp['name'], emp['position'], emp['embedding']
            embedding = np.frombuffer(base64.b64decode(embedding_blob), dtype='f')
            embeddings.append(embedding)
            employee_details.append((name, position))

        embeddings = np.array(embeddings, dtype='f')

        # Initialize FAISS index and search for the closest match
        num_dimensions = embeddings.shape[1]
        index = faiss.IndexFlatL2(num_dimensions)
        index.add(embeddings)
        k = 1  # Find the closest match
        distances, neighbours = index.search(np.array([target_embedding]), k)

        if len(neighbours) > 0 and neighbours[0][0] < len(employee_details):
            match_name, position = employee_details[neighbours[0][0]]
            return jsonify({"name": match_name, "position": position}), 200

        return jsonify({"message": "Tidak ada kecocokan ditemukan."}), 404

    except ValueError as ve:
        print(f"ValueError during identification: {str(ve)}")
        return jsonify({"message": f"ValueError: {str(ve)}"}), 400
    except Exception as e:
        print(f"Error during identification: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route('/employees', methods=['GET'])
def get_employees():
    try:
        employees_data = fetch_embeddings()
        if employees_data:
            return jsonify(employees_data), 200
        else:
            return jsonify({"message": "Tidak ada data karyawan."}), 404
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route('/employees-full', methods=['GET'])
def get_employees_full():
       try:
           with sqlite3.connect(db_path) as conn:
               cursor = conn.cursor()
               cursor.execute("SELECT id, name, position, embedding FROM employees")
               data = cursor.fetchall()

               employees_data = []
               for row in data:
                   id, name, position, embedding_blob = row
                   embedding_base64 = base64.b64encode(embedding_blob).decode('utf-8')
                   employees_data.append({
                       "id": id,
                       "name": name,
                       "position": position,
                       "embedding": embedding_base64
                   })

           return jsonify(employees_data), 200
       except Exception as e:
           return jsonify({"message": f"Error: {str(e)}"}), 500
    
# Fetch all employee info from the employee_info table
@app.route('/employees-info', methods=['GET'])
def get_employees_info():
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, name, position, image_base64 FROM employee_info")
            data = cursor.fetchall()

            employee_list = []
            for row in data:
                employee_id, name, position, image_base64 = row
                employee_list.append({
                    'employee_id': employee_id,
                    'name': name,
                    'position': position,
                    'image_base64': image_base64
                })

            return jsonify(employee_list), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

#menambah endpoint absen
@app.route('/record-attendance', methods=['POST'])
def record_attendance():
    data = request.json
    employee_name = data['name']
    image_capture = data['image']
    status = data.get('status', 'masuk')  # default status adalah masuk

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            if status == 'masuk':
                # Cek apakah sudah absen masuk hari ini
                cursor.execute("""
                    SELECT * FROM attendance 
                    WHERE employee_name = ? 
                    AND date = ? 
                    AND status = 'masuk'
                """, (employee_name, current_date))
                
                if cursor.fetchone():
                    return jsonify({
                        "message": f"Karyawan {employee_name} sudah melakukan absensi masuk hari ini"
                    }), 400
                
                # Catat absen masuk baru
                cursor.execute("""
                    INSERT INTO attendance (
                        employee_name, date, jam_masuk, jam_keluar, jam_kerja, 
                        image_capture, status
                    )
                    VALUES (?, ?, ?, NULL, NULL, ?, ?)
                """, (employee_name, current_date, current_time, image_capture, status))

            elif status == 'keluar':
                # Cek record absen masuk hari ini
                cursor.execute("""
                    SELECT id, jam_masuk 
                    FROM attendance 
                    WHERE employee_name = ? 
                    AND date = ? 
                    AND status = 'masuk'
                    AND jam_keluar IS NULL
                """, (employee_name, current_date))
                
                masuk_record = cursor.fetchone()
                
                if not masuk_record:
                    return jsonify({
                        "message": f"Tidak ditemukan absen masuk untuk {employee_name} hari ini"
                    }), 400
                
                # Hitung jam kerja
                record_id, jam_masuk = masuk_record
                jam_masuk_time = datetime.strptime(jam_masuk, '%H:%M:%S')
                jam_keluar_time = datetime.strptime(current_time, '%H:%M:%S')
                jam_kerja = str(jam_keluar_time - jam_masuk_time)

                # Update record dengan jam keluar dan jam kerja
                cursor.execute("""
                    UPDATE attendance 
                    SET jam_keluar = ?, 
                        jam_kerja = ?,
                        status = 'selesai'
                    WHERE id = ?
                """, (current_time, jam_kerja, record_id))

            conn.commit()
            return jsonify({
                "message": f"Absensi {status} berhasil dicatat untuk {employee_name}",
                "date": current_date,
                "time": current_time,
                "status": status
            }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

#mendapatkan data absensi
@app.route('/attendance-records', methods=['GET'])
def get_attendance_records():
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Ambil data absensi
            cursor.execute("""
                SELECT id, employee_name, date, jam_masuk, jam_keluar, jam_kerja, status, image_capture
                FROM attendance 
                ORDER BY date DESC, jam_masuk DESC
            """)

            records = cursor.fetchall()

            attendance_list = []
            for record in records:
                attendance_list.append({
                    'id': record[0],
                    'employee_name': record[1],
                    'date': record[2],
                    'jam_masuk': record[3],
                    'jam_keluar': record[4],
                    'jam_kerja': record[5],
                    'status': record[6],
                    'image_capture': record[7]
                })

            return jsonify(attendance_list), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

#endpoint data absen pe karyawan
@app.route('/attendance-records/<employee_name>', methods=['GET'])
def get_employee_attendance(employee_name):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, date, jam_masuk, jam_keluar, jam_kerja, status, image_capture 
                FROM attendance 
                WHERE employee_name = ? 
                ORDER BY date DESC, jam_masuk DESC
            """, (employee_name,))

            records = cursor.fetchall()

            attendance_list = []
            for record in records:
                attendance_list.append({
                    'id': record[0],
                    'date': record[1],
                    'jam_masuk': record[2],
                    'jam_keluar': record[3],
                    'jam_kerja': record[4],
                    'status': record[5],
                    'image_capture': record[6]
                })

            return jsonify(attendance_list), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

# Update employee in employee_info and employees table
@app.route('/update-employee/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    data = request.json
    name = data.get('name')
    position = data.get('position')
    image_base64 = data.get('image_base64', None)  # Optional field

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Update employee_info
            if image_base64:
                cursor.execute("""
                    UPDATE employee_info SET name = ?, position = ?, image_base64 = ?
                    WHERE employee_id = ?
                """, (name, position, image_base64, employee_id))
            else:
                cursor.execute("""
                    UPDATE employee_info SET name = ?, position = ?
                    WHERE employee_id = ?
                """, (name, position, employee_id))

            # Update employees (embedding table)
            cursor.execute("""
                UPDATE employees SET name = ?, position = ?
                WHERE id = ?
            """, (name, position, employee_id))
            
            conn.commit()
            return jsonify({"message": "Karyawan berhasil diperbarui!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

# Delete employee from both tables
@app.route('/delete-employee/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Delete from employee_info
            cursor.execute("DELETE FROM employee_info WHERE employee_id = ?", (employee_id,))

            # Delete from employees (embedding table)
            cursor.execute("DELETE FROM employees WHERE id = ?", (employee_id,))

            conn.commit()
            return jsonify({"message": "Karyawan berhasil dihapus!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True, host='0.0.0.0', port=5000)