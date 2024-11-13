import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from twilio.rest import Client  # Twilio client for sending SMS
from PyQt5.QtWidgets import QApplication, QFileDialog
from geopy.distance import geodesic

# Initialize Twilio client (replace with your actual credentials)
account_sid = 'ACcd5104988b78c92c52553d526198c15d'
auth_token = '70d1d854cd2aa7c23dc29db77ab63b0d'
twilio_phone_number = '+1 814 205 0137'
recipient_phone_number = '+917710879977'
client = Client(account_sid, auth_token)

# Function to send SMS notification with accumulated coordinates
def send_alert(accumulated_coords):
    try:
        # Format the message to include all accumulated coordinates
        message_body = "Potholes detected at the following locations:\n"
        for lat, lon in accumulated_coords:
            message_body += f"Latitude: {lat}, Longitude: {lon}\n"
        
        # Send the message via Twilio 
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print(f"Alert sent with SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Function to check if a pothole is within 20 meters of previously detected ones
def is_near_existing_pothole(lat, lon, existing_coords, threshold=20):
    for existing_lat, existing_lon in existing_coords:
        distance = geodesic((lat, lon), (existing_lat, existing_lon)).meters
        if distance < threshold:
            return True
    return False

# Load the CSV file containing GPS data
csv_file_path = '/Users/shreeshnadgouda/Desktop/Pothole_Detection/cleaned_lat_long_data.csv'
data = pd.read_csv(csv_file_path)

# Strip extra spaces and clean columns
data.columns = data.columns.str.strip()

# Extract coordinates where potholes are detected (PCI == 7)
coordinates = data[data['PCI'] == 7][['Latitude', 'Longitude']].values

# Load the YOLO model
model = YOLO("/Users/shreeshnadgouda/Downloads/best-2.pt")
class_names = model.names

# Initialize PyQt application for file selection
app = QApplication([])
file_path, _ = QFileDialog.getOpenFileName(None, "Select Image or Video", "", "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi)")
app.exit()

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Initialize variables for accumulating detections
accumulated_coords = []
frame_count = 0
notification_interval = 10  # Send notification every 10 seconds (approx.)
last_notification_time = time.time()

# Function to process detections and accumulate coordinates
def process_detections(results, latitude, longitude, img, detected_potholes):
    pothole_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0 and box.conf[0] > 0.3:  # Assuming class 0 is pothole
                pothole_detected = True
                x, y, w, h = box.xywh[0]
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), (int(y + h / 2))), (0, 0, 255), 2)
                cv2.putText(img, "Pothole", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # If a pothole is detected and it's not near an already detected pothole, accumulate the coordinates
    if pothole_detected and not is_near_existing_pothole(latitude, longitude, detected_potholes):
        detected_potholes.append((latitude, longitude))

# Process image or video
if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    img = cv2.imread(file_path)
    latitude, longitude = coordinates[0]  # For simplicity, assume first GPS coord
    results = model.predict(img)
    process_detections(results, latitude, longitude, img, accumulated_coords)
    cv2.imshow('Pothole Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif file_path.lower().endswith(('.mp4', '.avi')):
    cap = cv2.VideoCapture(file_path)
    frame_index = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % 3 == 0:  # Process every 3rd frame
            # Map current frame to GPS coordinates
            latitude, longitude = coordinates[frame_index % len(coordinates)]
            img = cv2.resize(img, (1020, 500))
            results = model.predict(img)
            process_detections(results, latitude, longitude, img, accumulated_coords)
            cv2.imshow('Pothole Detection', img)

            # Send notification every 10 seconds
            current_time = time.time()
            if current_time - last_notification_time >= notification_interval:
                if accumulated_coords:
                    send_alert(accumulated_coords)
                    accumulated_coords = []  # Reset the accumulated coordinates
                last_notification_time = current_time

            # Control playback speed and allow quitting with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unsupported file format. Please upload a valid image or video.")
