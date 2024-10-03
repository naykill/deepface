from flask import Flask, jsonify, request
from embedding_creation import create_embeddings
from faiss_storage import initialize_faiss
from face_recognition_webcam import recognize_face_from_webcam

app = Flask(__name__)

# Route for creating embeddings from image dataset
@app.route('/create-embeddings', methods=['POST'])
def create_embeddings_route():
    data = request.json
    image_folder = data.get('image_folder')
    if not image_folder:
        return jsonify({"error": "image_folder is required"}), 400

    result = create_embeddings(image_folder)
    return jsonify({"message": result})

# Route for initializing FAISS with embeddings
@app.route('/initialize-faiss', methods=['POST'])
def initialize_faiss_route():
    result = initialize_faiss()
    return jsonify({"message": result})

# Route for recognizing face from webcam
@app.route('/recognize-face', methods=['GET'])
def recognize_face_route():
    result = recognize_face_from_webcam()
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)
