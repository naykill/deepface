import cv2
import base64
import requests
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from gtts import gTTS
import os
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
import hashlib
from deepface import DeepFace

load_dotenv()

# Replace hardcoded values with environment variables
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC')
MQTT_USER = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
SERVER_URL = os.getenv('BACKEND_SERVER_URL')
CAMERA_CONFIG = os.getenv('CAMERA_CONFIGURE')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MQTT setup remains the same
MQTT_BROKER = MQTT_BROKER
MQTT_PORT = MQTT_PORT
MQTT_TOPIC = MQTT_TOPIC
MQTT_USER = MQTT_USER
MQTT_PASSWORD = MQTT_PASSWORD
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT broker")
    else:
        logger.error("Failed to connect to MQTT broker, Return code %d", rc)

mqtt_client.on_connect = on_connect
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def speak_text(text):
    tts = gTTS(text=text, lang='id')
    tts.save("greeting.mp3")
    os.system("mpg321 greeting.mp3")
    
def get_time_period():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "pagi"
    elif 12 <= hour < 15:
        return "siang"
    elif 15 <= hour < 18:
        return "sore"
    else:
        return "malam"

def get_last_attendance_from_api(employee_name):
    """
    Fetch the latest check-in and check-out times for the given employee from the API.
    """
    try:
        response = requests.get(
            f"{SERVER_URL}/attendance-records",
            timeout=8
        )
        response.raise_for_status()
        
        attendance_records = response.json()
        current_date = datetime.now().date()
        
        logger.debug(f"API Response for {employee_name}: {attendance_records}")
        
        is_unknown = employee_name.startswith('Unknown_')
        
        employee_records = [
            record for record in attendance_records
            if record['employee_name'] == employee_name
        ]
        
        if not employee_records:
            logger.info(f"No attendance records found for: {employee_name}")
            return None
            
        latest_record = employee_records[0]
        logger.debug(f"Latest record found: {latest_record}")
        
        if 'date' in latest_record and latest_record['date']:
            try:
                record_date = datetime.strptime(latest_record['date'], "%Y-%m-%d").date()
                check_in_time = latest_record.get('jam_masuk')
                check_out_time = latest_record.get('jam_keluar')
                
                if is_unknown and check_in_time:
                    last_check_in = datetime.combine(
                        record_date,
                        datetime.strptime(check_in_time, "%H:%M:%S").time()
                    )
                    time_since_last_record = datetime.now() - last_check_in
                    if time_since_last_record.total_seconds() > 300:
                        logger.info(f"Allowing new record for unknown person after 5 minutes")
                        return None
                else:
                    if record_date != current_date:
                        logger.info(f"Record date {record_date} differs from current date {current_date}")
                        return None
                
                status = latest_record.get('status', 'keluar')
                if is_unknown:
                    status = 'keluar'
                elif check_in_time and not check_out_time:
                    status = 'masuk'
                elif check_out_time:
                    status = 'keluar'
                
                return {
                    "check_in_time": check_in_time,
                    "check_out_time": check_out_time,
                    "status": status,
                    "date": record_date.strftime("%Y-%m-%d"),
                    "is_unknown": is_unknown
                }
                
            except ValueError as e:
                logger.error(f"Error parsing date/time from record: {e}")
                return None
        else:
            logger.warning(f"Date not found for {employee_name}, record: {latest_record}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Error processing attendance data: {e}")
        return None

def generate_face_hash(face_image_base64):
    return hashlib.md5(face_image_base64.encode()).hexdigest()
        
class FaceDetectionSystem:
    def __init__(self):
        self.unknown_faces = {}
        self.SERVER_URL = SERVER_URL
        self.CAPTURE_INTERVAL = 5
        self.CHECKOUT_INTERVAL = 20
        self.UNKNOWN_RECORD_INTERVAL = 300
        self.FRAME_SKIP = 2
        self.DETECTION_SCALE = 0.5
        self.attendance_records = {}
        self.cap = cv2.VideoCapture(CAMERA_CONFIG)
        self.detector_backend = "opencv"
        self.frame_count = 0
        self.last_capture_time = time.time()
    
    def _detect_faces(self, frame):
        """
        Detect faces using DeepFace
        """
        try:
            # Convert frame to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using DeepFace
            detected_faces = DeepFace.extract_faces(
                rgb_frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Extract face coordinates
            face_locations = []
            for face_dict in detected_faces:
                if 'facial_area' in face_dict:
                    area = face_dict['facial_area']
                    face_locations.append((
                        area['x'],
                        area['y'],
                        area['w'],
                        area['h']
                    ))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def _handle_unknown_person(self, face_data):
        current_time = time.time()
        face_hash = generate_face_hash(face_data)
        
        if face_hash in self.unknown_faces:
            last_detection = self.unknown_faces[face_hash]
            time_elapsed = current_time - last_detection
            
            if time_elapsed < 300:
                logger.info(f"Skipping capture for {face_hash[:8]} - detected {time_elapsed:.1f} seconds ago")
                return
        
        unknown_id = f"Unknown Person"
        
        try:
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": unknown_id,
                    "image": face_data,
                    "status": "masuk",
                },
                timeout=8
            )
            
            if response.status_code == 200:
                self.unknown_faces[face_hash] = current_time
                logger.info(f"Successfully recorded unknown person: {unknown_id}")
                mqtt_client.publish(MQTT_TOPIC, "Gate remains open for unknown person")
                speak_text(f"Tamu, Selamat {get_time_period()}.")
            else:
                logger.error(f"Failed to record unknown person: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to record unknown person: {e}")

    def _handle_face_recognition(self, face_data):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/identify-employee",
                json={"image": face_data},
                headers={'Content-Type': 'application/json'},
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                employee_name = data['name']
                
                if employee_name == "Unknown Person":
                    self._handle_unknown_person(face_data)
                    return
                
                current_time = time.time()
                last_attendance = get_last_attendance_from_api(employee_name)
                current_date = datetime.now().date()
                
                if employee_name not in self.attendance_records:
                    self.attendance_records[employee_name] = {
                        "status": "keluar" if not last_attendance else last_attendance["status"],
                        "check_in_time": None if not last_attendance else last_attendance["check_in_time"],
                        "last_detection": 0,
                        "last_checkout_time": None
                    }

                record = self.attendance_records[employee_name]

                if last_attendance:
                    record["status"] = last_attendance["status"]
                    record["check_in_time"] = last_attendance["check_in_time"]
                    if last_attendance["check_out_time"]:
                        record["last_checkout_time"] = datetime.strptime(
                            f"{last_attendance['date']} {last_attendance['check_out_time']}", 
                            "%Y-%m-%d %H:%M:%S"
                        )

                if current_time - record["last_detection"] >= self.CAPTURE_INTERVAL:
                    record["last_detection"] = current_time

                    if record["status"] == "keluar":
                        self._record_attendance(employee_name, face_data, "masuk")
                        record["status"] = "masuk"
                        record["check_in_time"] = datetime.now().strftime("%H:%M:%S")
                        logger.info(f"{employee_name} checked in")
                        mqtt_client.publish(MQTT_TOPIC, f"{employee_name} checked in")

                    elif record["status"] == "masuk":
                        if record["check_in_time"]:
                            check_in_time = datetime.strptime(record["check_in_time"], "%H:%M:%S").time()
                            check_in_datetime = datetime.combine(current_date, check_in_time)
                            time_since_checkin = (datetime.now() - check_in_datetime).total_seconds()
                            
                            if time_since_checkin >= self.CHECKOUT_INTERVAL:
                                self._record_attendance(employee_name, face_data, "keluar")
                                record["status"] = "keluar"
                                record["last_checkout_time"] = datetime.now()
                                record["check_in_time"] = None
                                logger.info(f"{employee_name} checked out after {time_since_checkin/60:.1f} minutes")
                                mqtt_client.publish(MQTT_TOPIC, f"{employee_name} checked out")
                        else:
                            logger.warning(f"Check-in time not found for {employee_name}, updating status to checked out")
                            self._record_attendance(employee_name, face_data, "keluar")
                            record["status"] = "keluar"
                            record["last_checkout_time"] = datetime.now()

        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")

    def _record_attendance(self, employee_name, image_base64, status, is_unknown=False):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": status
                },
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                time_period = data.get('period', get_time_period())
                
                if "warning" in data:
                    logger.warning(f"{employee_name} {status} - {time_period} - {data['warning']}")
                else:
                    logger.info(f"{employee_name} {status} - {time_period}")
                
                if not is_unknown:
                    if status == "masuk":
                        speak_text(f"Selamat {time_period} {employee_name}, selamat datang di ti leb, silakan {status}")
                    else:
                        speak_text(f"sampai jumpa {employee_name}, hati-hati di jalan")
                                   
            elif response.status_code == 400 and "early_checkout" in response.json().get("status", ""):
                logger.warning(f"Early checkout attempt for {employee_name} - Minimum working time not reached")
            else:
                logger.error(f"Failed to record attendance: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.FRAME_SKIP != 0:
            return frame

        faces = self._detect_faces(frame)

        current_time = time.time()
        if faces and (current_time - self.last_capture_time >= self.CAPTURE_INTERVAL):
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            
            _, buffer = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"Captured face for processing, size: {len(face_base64)} bytes")

            self._handle_face_recognition(face_base64)
            self.last_capture_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Face Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceDetectionSystem()
    system.run()