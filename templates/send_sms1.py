import cv2
import pandas as pd
from ultralytics import YOLO
from geopy.distance import geodesic
import requests
import subprocess
import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import time
import threading
from twilio.rest import Client

# Server and threshold configuration
SERVER_URL = 'http://127.0.0.1:5000/update_potholes'
PROXIMITY_THRESHOLD_METERS = 100  # Detection threshold
NOTIFICATION_INTERVAL = 10  # Notify every 10 seconds to avoid spamming

# Twilio configuration
account_sid = 'AC4259145fc30a700a3ffbb1ab2ed5818b'
auth_token = '26a06621561d4a4b1963b55e981cc580'
twilio_phone_number = '+1 267 507 9760'
recipient_phone_number = '+917710879977'
client = Client(account_sid, auth_token)

# Track sent coordinates to avoid redundant notifications
all_detected_coords = set()
accumulated_coords = []
notification_lock = threading.Lock()
last_notification_time = time.time()

# Function to send pothole data to Flask server
def send_pothole_data(lat, lon):
    data = {'lat': lat, 'lon': lon, 'detection_active': True}
    try:
        response = requests.post(SERVER_URL, json=data, timeout=5)
        if response.status_code == 200:
            print("Data sent successfully:", response.json())
        else:
            print(f"Error: Received status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Function to send SMS notifications in real-time, every 10 seconds
def send_cumulative_sms_alert():
    global accumulated_coords, last_notification_time
    current_time = time.time()
    if current_time - last_notification_time >= NOTIFICATION_INTERVAL:
        with notification_lock:
            if accumulated_coords:
                # Filter out coordinates that are within 5 meters to reduce redundancy
                unique_coords = filter_nearby_potholes(accumulated_coords)
                if unique_coords:
                    message_body = "Detected potholes at the following unique locations:\n"
                    for lat, lon in unique_coords:
                        message_body += f"Latitude: {lat}, Longitude: {lon}\n"
                        all_detected_coords.add((lat, lon))  # Mark these as sent
                    
                    try:
                        message = client.messages.create(
                            body=message_body,
                            from_=twilio_phone_number,
                            to=recipient_phone_number
                        )
                        print(f"Alert sent with SID: {message.sid}")
                    except Exception as e:
                        print(f"Failed to send alert: {e}")
                
                accumulated_coords = []  # Reset accumulated coordinates after sending
        last_notification_time = current_time  # Update last notification time

# Function to send SMS notifications in batches of unique coordinates
def send_sms_alert_in_batches(all_coords, batch_size=10):
    unique_coords = list(set(all_coords))  # Ensure uniqueness by using a set

    if not unique_coords:
        print("No potholes detected; no SMS will be sent.")
        return

    # Construct message body in batches, ensuring each message stays within SMS character limits
    messages = []
    current_message = "Detected potholes at the following unique locations:\n"
    max_message_length = 1600

    for i in range(0, len(unique_coords), batch_size):
        batch = unique_coords[i:i + batch_size]
        batch_message = ""
        for idx, (lat, lon) in enumerate(batch, 1):
            entry = f"{idx}. Latitude: {lat}, Longitude: {lon}\n"
            batch_message += entry
        
        # Ensure each batch message does not exceed character limit
        if len(current_message) + len(batch_message) > max_message_length:
            messages.append(current_message)
            current_message = "Detected potholes at the following unique locations:\n" + batch_message
        else:
            current_message += batch_message

    # Append any remaining message
    if current_message.strip() != "Detected potholes at the following unique locations:\n":
        messages.append(current_message)

    # Send each batch as a separate SMS
    try:
        for i, message_content in enumerate(messages):
            message = client.messages.create(
                body=message_content,
                from_=twilio_phone_number,
                to=recipient_phone_number
            )
            print(f"SMS batch {i+1} sent with SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

# Function to filter out nearby potholes within a specified threshold
def filter_nearby_potholes(coords, threshold=5):
    filtered_coords = []
    for coord in coords:
        if coord not in all_detected_coords and all(geodesic(coord, c).meters > threshold for c in filtered_coords):
            filtered_coords.append(coord)
            all_detected_coords.add(coord)  # Mark as sent to avoid duplicates
    return filtered_coords

# Function to convert GPS degrees, minutes, seconds to decimal format
def dms_to_decimal(degrees, minutes, seconds, direction):
    seconds = re.sub(r'[^0-9.]', '', seconds)  # Remove any non-numeric characters
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    return -decimal if direction in ['S', 'W'] else decimal

# Function to extract GPS data from video metadata using exiftool
def extract_gps_from_video(video_file_path):
    process = subprocess.Popen(['exiftool', video_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    metadata = out.decode('utf-8').split('\n')
    gps_data = {}

    for line in metadata:
        if 'GPS Latitude' in line:
            gps_data['Latitude'] = line.split(': ')[1].strip()
        if 'GPS Longitude' in line:
            gps_data['Longitude'] = line.split(': ')[1].strip()

    if 'Latitude' in gps_data and 'Longitude' in gps_data:
        lat_match = re.match(r'(\d+) deg (\d+)\x27 (\d+\.\d+\") ([N|S])', gps_data['Latitude'])
        lon_match = re.match(r'(\d+) deg (\d+)\x27 (\d+\.\d+\") ([E|W])', gps_data['Longitude'])

        if lat_match and lon_match:
            lat_degrees, lat_minutes, lat_seconds, lat_direction = lat_match.groups()
            lon_degrees, lon_minutes, lon_seconds, lon_direction = lon_match.groups()
            latitude = dms_to_decimal(lat_degrees, lat_minutes, lat_seconds, lat_direction)
            longitude = dms_to_decimal(lon_degrees, lon_minutes, lon_seconds, lon_direction)
            return latitude, longitude
    return None

# Load CSV containing GPS data (pothole locations)
csv_file_path = '/Users/shreeshnadgouda/Desktop/Pothole_Detection/cleaned_lat_long_data.csv'
data = pd.read_csv(csv_file_path)
data.columns = data.columns.str.strip()

# Filter potholes near initial GPS coordinates from video
def filter_nearby_potholes_initial(initial_lat, initial_lon, threshold=PROXIMITY_THRESHOLD_METERS):
    nearby_potholes = []
    for _, row in data.iterrows():
        pothole_lat = row['Latitude']
        pothole_lon = row['Longitude']
        distance = geodesic((initial_lat, initial_lon), (pothole_lat, pothole_lon)).meters
        if distance <= threshold:
            nearby_potholes.append((pothole_lat, pothole_lon))
    return nearby_potholes

# Select file via PyQt dialog (for image or video file)
app = QApplication([])
file_path, _ = QFileDialog.getOpenFileName(None, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
app.exit()

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Try to get initial GPS coordinates; handle case where metadata is unavailable
gps_metadata = extract_gps_from_video(file_path)
metadata_available = gps_metadata is not None

if metadata_available:
    initial_lat, initial_lon = gps_metadata
    print(f"Initial video coordinates: Latitude = {initial_lat}, Longitude = {initial_lon}")

    # Send initial location to server once
    nearby_potholes = filter_nearby_potholes_initial(initial_lat, initial_lon)
    all_detected_coords.update(nearby_potholes)
    for lat, lon in nearby_potholes:
        send_pothole_data(lat, lon)
else:
    print("No GPS metadata found in the video; proceeding without real-time mapping.")

# Initialize YOLO model
model = YOLO("/Users/shreeshnadgouda/Desktop/Pothole_Detection/best-3.pt")

# Process video
cap = cv2.VideoCapture(file_path)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    results = model.predict(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0 and box.conf[0] > 0.3:
                x, y, w, h = box.xywh[0]
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)
                cv2.putText(img, "Pothole", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Accumulate pothole coordinates only if metadata is available
                if metadata_available:
                    with notification_lock:
                        accumulated_coords.append((initial_lat, initial_lon))
                    send_pothole_data(initial_lat, initial_lon)

    send_cumulative_sms_alert()  # Send SMS alerts during video processing
    cv2.imshow('Pothole Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Send consolidated SMS in batches with all detected pothole locations if metadata was available
if metadata_available:
    send_sms_alert_in_batches(list(all_detected_coords), batch_size=10)
