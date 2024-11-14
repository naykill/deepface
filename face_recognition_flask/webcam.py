import cv2
import base64
import requests
import numpy as np
import time
from datetime import datetime

# Initialize webcam
cap = cv2.VideoCapture("http://172.254.0.124:2000/video")  # Change to 1 for external webcam
if not cap.isOpened():
    print("Error: Cannot open webcam. Make sure the camera is connected and working.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SERVER_URL = "http://172.254.2.153:5000"

# ESP32 URLs for controlling the gate
ESP32_URL_OPEN = "http://172.254.2.78/open-gate"
ESP32_URL_CLOSE = "http://172.254.2.78/close-gate"

# Set capture interval (5 seconds)
capture_interval = 5
start_time = time.time()

# Variables for tracking attendance
attendance_status = {}  # Track check-in/check-out status for each employee
current_date = datetime.now().strftime('%Y-%m-%d')

def reset_attendance_record():
    """Reset attendance records at the start of a new day"""
    global attendance_status, current_date
    new_date = datetime.now().strftime('%Y-%m-%d')
    if new_date != current_date:
        attendance_status.clear()
        current_date = new_date

def handle_attendance(employee_name, image_base64):
    """Handle the attendance logic for an employee"""
    try:
        # Check if employee has checked in today
        if employee_name not in attendance_status:
            # Attempt to check in
            attendance_response = requests.post(
                f"{SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": "masuk"
                }
            )
        else:
            # If already checked in, attempt to check out
            attendance_response = requests.post(
                f"{SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": "keluar"
                }
            )
        
        if attendance_response.status_code == 200:
            response_data = attendance_response.json()
            status = response_data.get('status', '')
            
            if status == 'masuk':
                attendance_status[employee_name] = 'checked_in'
                print(f"Check-in successful: {employee_name}")
            elif status == 'selesai':
                attendance_status.pop(employee_name, None)
                print(f"Check-out successful: {employee_name}")
                
            print(f"Date: {response_data['date']}")
            print(f"Time: {response_data['time']}")
            print(f"Period: {response_data['time_period']}")
            return True
            
        else:
            error_msg = attendance_response.json().get('message', 'Unknown error')
            print(f"Attendance recording failed: {error_msg}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False

# Add flag for gate status
gate_opened = False

while True:
    reset_attendance_record()  # Reset attendance records if day changes
    
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam.")
        break

    # Detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # If face is detected, capture face
    if len(faces) > 0:
        current_time = time.time()
        if current_time - start_time >= capture_interval:
            for (x, y, w, h) in faces:
                # Extract face from frame
                face = frame[y:y+h, x:x+w]

                # Convert face to base64
                _, buffer = cv2.imencode('.jpg', face)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send image to server for identification
                try:
                    response = requests.post(
                        f"{SERVER_URL}/identify-employee", 
                        json={"image": image_base64},
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        employee_name = data['name']
                        
                        # Handle attendance and get success status
                        attendance_success = handle_attendance(employee_name, image_base64)
                        
                        print(f"Welcome {data['name']} - {data['position']}")
                        print(f"Confidence: {data['confidence']:.2f}")

                        # Open gate only if attendance was successful
                        if attendance_success and not gate_opened:
                            try:
                                esp_response = requests.get(ESP32_URL_OPEN)
                                if esp_response.status_code == 200:
                                    print("Gate opened successfully!")
                                    gate_opened = True
                            except requests.exceptions.RequestException as e:
                                print(f"Error connecting to ESP32: {e}")
                    else:
                        print(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    print(f"Error connecting to server: {e}")

            start_time = current_time

    else:
        # Close gate if no face is detected and gate is open
        if gate_opened:
            try:
                esp_response = requests.get(ESP32_URL_CLOSE)
                if esp_response.status_code == 200:
                    print("Gate closed successfully!")
                    gate_opened = False
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to ESP32: {e}")

    # Display webcam frame with face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()