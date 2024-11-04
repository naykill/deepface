import cv2
import base64
import requests
import numpy as np
import time
from datetime import datetime
import threading
from queue import Queue
import logging
import torch
from ultralytics import YOLO
from scipy.spatial import distance
from skimage.feature import local_binary_pattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiSpoofing:
    def __init__(self):
        self.LBP_POINTS = 8
        self.LBP_RADIUS = 1
        self.GRID_SIZE = 8
        self.THRESHOLD = 0.35  # Adjust this threshold based on testing
        
    def _calculate_lbp_histogram(self, image):
        """Calculate LBP histogram for the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, self.LBP_POINTS, self.LBP_RADIUS, method='uniform')
        
        # Calculate histogram for each grid cell
        hist_list = []
        h, w = gray.shape
        grid_h, grid_w = h // self.GRID_SIZE, w // self.GRID_SIZE
        
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                cell = lbp[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                hist, _ = np.histogram(cell, bins=self.LBP_POINTS + 2, range=(0, self.LBP_POINTS + 2))
                hist = hist.astype('float') / np.sum(hist)
                hist_list.extend(hist)
                
        return np.array(hist_list)

    def _calculate_texture_score(self, image):
        """Calculate texture variation score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(gradient_magnitude)

    def _calculate_color_variation(self, image):
        """Calculate color variation score"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return np.std(saturation)

    def check_liveness(self, face_image):
        """Check if the face is real or spoofed"""
        # Resize image for consistent processing
        face_image = cv2.resize(face_image, (128, 128))
        
        # Calculate various features
        lbp_hist = self._calculate_lbp_histogram(face_image)
        texture_score = self._calculate_texture_score(face_image)
        color_var = self._calculate_color_variation(face_image)
        
        # Combined score (you may need to adjust weights based on testing)
        texture_weight = 0.4
        color_weight = 0.3
        lbp_weight = 0.3
        
        normalized_texture = texture_score / 100  # Normalize to 0-1 range
        normalized_color = color_var / 255  # Normalize to 0-1 range
        normalized_lbp = np.mean(lbp_hist)  # Already normalized
        
        combined_score = (texture_weight * normalized_texture +
                         color_weight * normalized_color +
                         lbp_weight * normalized_lbp)
        
        return combined_score > self.THRESHOLD

class FaceDetectionSystem:
    def __init__(self):
        # Configuration
        self.SERVER_URL = "http://172.254.2.153:5000"
        self.ESP32_URL_OPEN = "http://172.254.2.78/open-gate"
        self.ESP32_URL_CLOSE = "http://172.254.2.78/close-gate"
        self.CAPTURE_INTERVAL = 5
        self.FRAME_SKIP = 2  # Process every nth frame
        self.CONF_THRESHOLD = 0.5  # Confidence threshold for detection
        
        # Initialize camera with RTSP
        self.cap = cv2.VideoCapture("http://172.254.1.122:4747/video")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        # Initialize YOLOv5-face model
        self.model = self._load_model()
        
        # Initialize anti-spoofing
        self.anti_spoofing = AntiSpoofing()
        
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

    def _load_model(self):
        """Load YOLOv5-face model"""
        try:
            # Load YOLOv5-face model (nano version for lightweight)
            model = YOLO('yolov8n-face.pt')
            
            # Configure model for inference
            model.conf = self.CONF_THRESHOLD  # Confidence threshold
            model.iou = 0.45  # NMS IoU threshold
            
            # Optimize model for Jetson Nano
            if torch.cuda.is_available():
                model.cuda()  # Use GPU if available
            model.eval()  # Set to evaluation mode
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _process_api_requests(self):
        """Background thread for handling API requests"""
        while True:
            try:
                face_data, is_real = self.api_queue.get()
                if is_real:
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
        """Control the gate state"""
        if should_open != self.gate_opened:
            try:
                url = self.ESP32_URL_OPEN if should_open else self.ESP32_URL_CLOSE
                requests.get(url, timeout=2)
                self.gate_opened = should_open
                logger.info(f"Gate {'opened' if should_open else 'closed'}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Gate control error: {e}")

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

        # Convert frame for YOLOv5
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model(img)
        
        # Process detections
        if len(results.pred[0]) > 0:  # If faces detected
            current_time = time.time()
            
            # Get the detection with highest confidence
            det = results.pred[0]
            best_det = det[det[:, 4].argmax()]
            
            if current_time - self.last_capture_time >= self.CAPTURE_INTERVAL:
                x1, y1, x2, y2 = map(int, best_det[:4])
                face = frame[y1:y2, x1:x2]
                
                # Check for spoofing
                is_real = self.anti_spoofing.check_liveness(face)
                
                # Encode face image if real
                if is_real:
                    _, buffer = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Queue face data for processing if queue is not full
                    if self.api_queue.empty():
                        self.api_queue.put((face_base64, True))
                
                self.last_capture_time = current_time

            # Draw detections on frame with color based on liveness
            for det in results.pred[0]:
                x1, y1, x2, y2, conf = map(int, det[:5])
                face = frame[y1:y2, x1:x2]
                is_real = self.anti_spoofing.check_liveness(face)
                
                # Set color based on liveness (Green for real, Red for spoof)
                color = (0, 255, 0) if is_real else (0, 0, 255)
                status = "Real" if is_real else "Spoof"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{status} {conf:.2f}', (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
                if len(self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pred[0]) == 0:
                    self._control_gate(False)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceDetectionSystem()
    system.run()