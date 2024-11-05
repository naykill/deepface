import cv2
import base64
import requests
import numpy as np
import time
import logging
from datetime import datetime
import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectionSystem:
    def __init__(self):
        # Configuration
        self.SERVER_URL = "http://172.254.2.153:5000"
        self.CAPTURE_INTERVAL = 5
        self.CHECKOUT_INTERVAL = 600  # 10 minutes in seconds
        self.FRAME_SKIP = 2  # Process every nth frame
        self.DETECTION_SCALE = 0.5  # Scale down factor for face detection
        self.unknown_cooldown = 300  # 5 minutes cooldown for unknown person detection
        self.last_unknown_detection = 0

        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Default camera
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # State variables
        self.attendance_records = {}  # Store check-in times and status for each person
        self.last_capture_time = time.time()
        self.frame_count = 0
    
    def _handle_face_recognition(self, face_data):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/identify-employee",
                json={"image": face_data},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                employee_name = data['name']
                confidence = data.get('confidence', 0)
                
                current_time = time.time()
                
                # Handle Unknown Person differently
                if employee_name == "Unknown Person":
                    # Check if enough time has passed since last unknown detection
                    if current_time - self.last_unknown_detection >= self.unknown_cooldown:
                        self._record_attendance(employee_name, face_data)
                        self.last_unknown_detection = current_time
                        self._control_gate(False)  # Don't open gate for unknown persons
                else:
                    # Known employee logic
                    if employee_name not in self.attendance_recorded:
                        self._record_attendance(employee_name, face_data)
                        self._control_gate(True)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")
    
    def _check_checkout_eligibility(self, employee_name):
        """Check if employee is eligible for checkout based on time elapsed since check-in"""
        if employee_name in self.attendance_records:
            record = self.attendance_records[employee_name]
            if record['status'] == 'masuk':
                time_elapsed = time.time() - record['check_in_time']
                return time_elapsed >= self.CHECKOUT_INTERVAL
        return False

    def _send_to_api(self, image_base64, status="masuk"):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/identify-employee",
                json={"image": image_base64},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                employee_name = data.get("name")
                confidence = data.get("confidence")
                
                if employee_name == "Unknown Person":
                    self._record_attendance("Unknown Person", image_base64, "masuk")
                    logger.info("Unknown person detected and recorded")
                else:
                    if self._check_checkout_eligibility(employee_name):
                        status = "keluar"
                    self._record_attendance(employee_name, image_base64, status)
                    
            else:
                logger.error("Failed to identify employee")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")

            
    def _record_attendance(self, employee_name, image_base64, status):
        """Record attendance for an employee with specified status (masuk/keluar)"""
        try:
            current_time = time.time()
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": status
                },
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"{status.capitalize()} recorded for {employee_name}")
                # For check-in
            if status == "masuk":
                if employee_name not in self.attendance_records:
                    self.attendance_records[employee_name] = {
                        'status': 'masuk',
                        'check_in_time': current_time
                    }
            
            # For check-out
            elif status == "keluar" and employee_name in self.attendance_records:
                if self.attendance_records[employee_name]['status'] == 'masuk':
                    working_time = current_time - self.attendance_records[employee_name]['check_in_time']
                    
                    # Only proceed with checkout if minimum time has passed
                    if working_time >= self.CHECKOUT_INTERVAL:
                        response = requests.post(
                            f"{self.SERVER_URL}/record-attendance",
                            json={
                                "name": employee_name,
                                "image": image_base64,
                                "status": status
                            },
                            timeout=5
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"Checkout recorded for {employee_name}. Working time: {timedelta(seconds=working_time)}")
                            # Reset the record after successful checkout
                            del self.attendance_records[employee_name]
                        else:
                            logger.warning(f"Checkout recording failed: {response.text}")
                    return
            # Record the attendance
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": status
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"{status.capitalize()} recorded for {employee_name}")
            else:
                logger.warning(f"Attendance recording failed: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Attendance recording error: {e}")

    def process_frame(self, frame):
        """Process a single frame for face detection"""
        # Skip frames to reduce processing load
        self.frame_count += 1
        if self.frame_count % self.FRAME_SKIP != 0:
            return frame

        # Scale down frame for face detection
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (int(width * self.DETECTION_SCALE), int(height * self.DETECTION_SCALE)))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Scale face coordinates back to original size
        faces = [(int(x/self.DETECTION_SCALE), int(y/self.DETECTION_SCALE),
                  int(w/self.DETECTION_SCALE), int(h/self.DETECTION_SCALE)) for (x,y,w,h) in faces]

        current_time = time.time()
        if faces and (current_time - self.last_capture_time >= self.CAPTURE_INTERVAL):
            x, y, w, h = faces[0]  # Process only the first detected face
            face = frame[y:y+h, x:x+w]

            # Encode face image to base64
            _, buffer = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send face data to API
            self._send_to_api(face_base64, "masuk")  # Default to "masuk", logic in _send_to_api will determine if it should be "keluar"

            self.last_capture_time = current_time

        # Draw rectangles around detected faces with status
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

    def run(self):
        """Main processing loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame
                cv2.imshow('Face Detection', processed_frame)

                # Check for exit command
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceDetectionSystem()
    system.run()
