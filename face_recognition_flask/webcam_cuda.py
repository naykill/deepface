import cv2
import base64
import requests
import numpy as np
import time
from datetime import datetime
import threading
from queue import Queue
import logging
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectionSystem:
    def __init__(self):
        # Configuration
        self.SERVER_URL = "http://172.254.2.153:5000"
        self.MQTT_BROKER = "172.254.2.153"  # Jetson Nano's IP
        self.MQTT_PORT = 1883
        self.MQTT_TOPIC_OPEN = "gate/open"
        self.MQTT_TOPIC_CLOSE = "gate/close"
        self.CAPTURE_INTERVAL = 5
        self.FRAME_SKIP = 2  # Process every nth frame
        self.DETECTION_SCALE = 0.5  # Scale down factor for face detection
        
        # Initialize camera with RTSP
        self.cap = cv2.VideoCapture("http://172.254.0.124:2000/video")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # State variables
        self.attendance_recorded = set()
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.gate_opened = False
        self.last_capture_time = time.time()
        self.frame_count = 0
        
        # Threading setup
        self.api_queue = Queue(maxsize=1)  # Only queue latest detection
        self.api_thread = threading.Thread(target=self._process_api_requests, daemon=True)
        self.api_thread.start()
        
        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
        self.mqtt_client.loop_start()

    def _process_api_requests(self):
        """Background thread for handling API requests"""
        while True:
            try:
                face_data = self.api_queue.get()
                self._handle_face_recognition(face_data)
            except Exception as e:
                logger.error(f"API processing error: {e}")

    def _handle_face_recognition(self, face_data):
        """Handle face recognition and API calls"""
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
                
                if employee_name not in self.attendance_recorded:
                    self._record_attendance(employee_name, face_data)
                    self._control_gate(True)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")

    def _record_attendance(self, employee_name, image_data):
        """Record attendance for an employee"""
        try:
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_data,
                    "status": "masuk"
                },
                timeout=5
            )
            
            if response.status_code == 200:
                self.attendance_recorded.add(employee_name)
                logger.info(f"Attendance recorded for {employee_name}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Attendance recording error: {e}")

    def _control_gate(self, should_open):
        """Control the gate state using MQTT"""
        if should_open != self.gate_opened:
            try:
                topic = self.MQTT_TOPIC_OPEN if should_open else self.MQTT_TOPIC_CLOSE
                self.mqtt_client.publish(topic, "1" if should_open else "0")
                self.gate_opened = should_open
                logger.info(f"Gate {'opened' if should_open else 'closed'} via MQTT")
            except Exception as e:
                logger.error(f"Gate control error via MQTT: {e}")

    def _reset_attendance_if_needed(self):
        """Reset attendance records at midnight"""
        current = datetime.now().strftime('%Y-%m-%d')
        if current != self.current_date:
            self.attendance_recorded.clear()
            self.current_date = current

    def process_frame(self, frame):
        """Process a single frame for face detection"""
        # Skip frames to reduce processing load
        self.frame_count += 1
        if self.frame_count % self.FRAME_SKIP != 0:
            return frame

        # Scale down frame for face detection
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (int(width * self.DETECTION_SCALE), 
                                       int(height * self.DETECTION_SCALE)))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        # Scale face coordinates back to original size
        faces = [(int(x/self.DETECTION_SCALE), int(y/self.DETECTION_SCALE),
                 int(w/self.DETECTION_SCALE), int(h/self.DETECTION_SCALE)) for (x,y,w,h) in faces]

        current_time = time.time()
        if faces and (current_time - self.last_capture_time >= self.CAPTURE_INTERVAL):
            x, y, w, h = faces[0]  # Process only the first detected face
            face = frame[y:y+h, x:x+w]
            
            # Encode face image
            _, buffer = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 80])
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Queue face data for processing if queue is not full
            if self.api_queue.empty():
                self.api_queue.put(face_base64)
            
            self.last_capture_time = current_time

        # Draw rectangles around faces
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

                self._reset_attendance_if_needed()
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Webcam', processed_frame)
                
                # Check for exit command
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # Close gate if no faces detected
                if not any(cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for _ in [0]):
                    self._control_gate(False)

        finally:
            self.cap.release()
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceDetectionSystem()
    system.run()

