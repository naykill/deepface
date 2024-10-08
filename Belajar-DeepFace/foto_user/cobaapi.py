from flask import Flask, request, jsonify

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Data dummy untuk keperluan API
users = [
    {"id": 1, "name": "John Doe", "age": 30},
    {"id": 2, "name": "Jane Doe", "age": 25}
]

# Endpoint GET untuk mengambil semua data user
@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(users)

# Endpoint GET untuk mengambil user berdasarkan ID
@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        return jsonify(user)
    else:
        return jsonify({"message": "User not found"}), 404

# Endpoint POST untuk menambahkan user baru
@app.route('/users', methods=['POST'])
def add_user():
    new_user = request.get_json()  # Mengambil data dari body request
    new_user['id'] = len(users) + 1  # Menambahkan ID secara otomatis
    users.append(new_user)
    return jsonify(new_user), 201  # Status 201 menandakan berhasil membuat resource baru

# Endpoint PUT untuk memperbarui data user
@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        data = request.get_json()
        user.update(data)  # Memperbarui data user yang ada
        return jsonify(user)
    else:
        return jsonify({"message": "User not found"}), 404

# Endpoint DELETE untuk menghapus user
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    global users
    users = [user for user in users if user["id"] != user_id]
    return jsonify({"message": "User deleted"}), 200

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
