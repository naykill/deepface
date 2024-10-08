import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from deepface import DeepFace
import faiss
import json
import ast

# Folder and file setup for saving data
FOTO_FOLDER = 'foto_user'
CSV_FILE = 'face_embeddings.csv'

# Create folder if not exists
if not os.path.exists(FOTO_FOLDER):
    os.makedirs(FOTO_FOLDER)

# Function to save data to CSV
def simpan_data(nama, posisi, foto_path, embedding):
    # Check if CSV exists
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=["Nama", "Posisi", "Embedding"])
        df.to_csv(CSV_FILE, index=False)
    
    # Read existing data
    df = pd.read_csv(CSV_FILE)
    
    # Add new data
    new_data = pd.DataFrame({
        "Nama": [nama],
        "Posisi": [posisi],
        "Embedding": [embedding]  # Save embedding directly as a list
    })
    
    # Use pd.concat to add new data
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the data back to the CSV file
    df.to_csv(CSV_FILE, index=False)

# Function to get embeddings from the image using DeepFace
def generate_embedding(foto_path):
    model_name = "Facenet"
    detector_backend = "opencv"
    embedding = DeepFace.represent(
        img_path=foto_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False  # Set to False to handle images without clear faces
    )
    return embedding[0]["embedding"]  # This should already be a list

# Function to safely parse embeddings back to lists from CSV
def parse_embedding(embedding_str):
    if pd.isna(embedding_str):  # Check if the value is NaN
        return []  # Return an empty list or handle it as needed
    return ast.literal_eval(embedding_str)

# Function to export FAISS index to JSON
def faiss_to_json(index, df):
    embeddings = index.reconstruct_n(0, index.ntotal)
    embeddings_list = embeddings.tolist()

    data_json = []
    for i, row in df.iterrows():
        data_json.append({
            "Nama": row["Nama"],
            "Posisi": row["Posisi"],
            "Embedding": embeddings_list[i]
        })

    json_data = json.dumps(data_json, indent=4)
    return json_data

# Function to crop face from the image
def crop_face(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Crop the first detected face (if any)
    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]  # Return the cropped face

    return None  # If no face is detected

# Streamlit interface
st.title("Employee Registration and Photo Submission")
st.write("Please enter your name, position, and capture or upload a photo.")

# Input fields for name and position
nama = st.text_input("Name")
posisi = st.text_input("Position")

# Camera input for capturing photo
foto = st.camera_input("Capture Photo")

# Save data and photo when button is clicked
if st.button("Save Data"):
    if nama and posisi and foto:
        # Convert to a format suitable for OpenCV
        file_bytes = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Crop the face from the image
        cropped_face = crop_face(image)
        
        if cropped_face is not None:
            # Save the cropped photo
            foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(foto_path, cropped_face)  # Save the cropped face
            
            # Generate embedding
            embedding = generate_embedding(foto_path)
            
            # Save data to CSV
            simpan_data(nama, posisi, foto_path, embedding)  # Save embedding directly
            
            st.success(f"Data successfully saved! Photo saved at: {foto_path}")
        else:
            st.error("No face detected in the captured photo.")
    else:
        st.error("Please fill out all fields (name, position, and photo).")

# Upload photo option for registering with a file
st.header("Upload Employee Photo")
uploaded_foto = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_foto is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_foto)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Crop the face from the uploaded image
    cropped_face = crop_face(image_cv)
    
    if cropped_face is not None:
        # Save the cropped photo
        foto_path = os.path.join(FOTO_FOLDER, f"{nama}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(foto_path, cropped_face)  # Save the cropped face
        
        # Display the uploaded image
        st.image(cropped_face, caption=f"Cropped photo for {nama}", use_column_width=True)
        
        # Generate embedding and save
        embedding = generate_embedding(foto_path)
        simpan_data(nama, posisi, foto_path, embedding)

        st.success(f"Data successfully saved with uploaded photo at: {foto_path}")
    else:
        st.error("No face detected in the uploaded photo.")

# Button to export FAISS index to JSON
if st.button("Export FAISS Index as JSON"):
    # Read CSV data
    df = pd.read_csv(CSV_FILE)
    
    # Convert embedding from string to list
    df['Embedding'] = df['Embedding'].apply(parse_embedding)  # Handle NaN values here
    embeddings = np.array(df['Embedding'].tolist(), dtype='f')

    # Initialize FAISS
    num_dimensions = 128  # Dimensionality of the embedding (for Facenet)
    index = faiss.IndexFlatL2(num_dimensions)

    # Add embeddings to the FAISS index
    index.add(embeddings)

    # Export FAISS index to JSON
    json_data = faiss_to_json(index, df)
    
    # Display JSON
    st.json(json.loads(json_data))
