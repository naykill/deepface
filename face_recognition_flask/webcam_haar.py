import cv2
import base64
import requests
import numpy as np
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectionSystem:
    def __init__(self):
        # Configuration
        self.SERVER_URL = "http://172.254.2.153:5000"
        self.CAPTURE_INTERVAL = 5
        self.FRAME_SKIP = 2  # Process every nth frame
        self.DETECTION_SCALE = 0.5  # Scale down factor for face detection

        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Default camera
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # State variables
        self.attendance_recorded = set()
        self.last_capture_time = time.time()
        self.frame_count = 0

    def _send_to_api(self, image_base64, status):
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
                logger.info(f"Identified {employee_name} with confidence {confidence}")
                self._record_attendance(employee_name, image_base64, status)
            elif response.status_code == 404:  # Unrecognized person
                self._record_attendance("Unknown Person", image_base64, status)
            else:
                logger.warning("Person not recognized.")
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")

    def _record_attendance(self, employee_name, image_base64, status):
        """Record attendance for an employee with specified status (masuk/keluar)"""
        try:
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
                if status == "masuk":
                    self.attendance_recorded.add(employee_name)  # Mark as checked in
                elif status == "keluar":
                    self.attendance_recorded.discard(employee_name)  # Mark as checked out
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

            # Determine attendance status (check-in or check-out)
            status = "keluar" if faces[0] in self.attendance_recorded else "masuk"

            # Send face data to API
            self._send_to_api(face_base64, status)

            self.last_capture_time = current_time

        # Draw rectangles around detected faces
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
