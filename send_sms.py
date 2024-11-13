import cv2
import pandas as pd
from ultralytics import YOLO
from twilio.rest import Client  # Twilio client for sending SMS
from geopy.distance import geodesic  # For calculating distance between coordinates
from PyQt5.QtWidgets import QApplication, QFileDialog
import subprocess
import re
import numpy as np
import requests  # Added to send data to Flask

# Initialize Twilio client (replace with your actual credentials)
account_sid = 'AC0a1cc98a3ceae85afec42ea22f0ccc6b'
auth_token = 'b9bf85971dc783912f67e09cfe8e67bb'
twilio_phone_number = '+1 864 385 2914'
recipient_phone_number = '+917710879977'

client = Client(account_sid, auth_token)


# Set to keep track of already sent coordinates
sent_coords = set()

# Function to send pothole data to Flask server
def send_pothole_data(lat, lon):
    url = 'http://127.0.0.1:5000/update_potholes'
    data = {'lat': lat, 'lon': lon}
    try:
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            try:
                print(response.json())  # Log server response if it's valid JSON
            except ValueError:
                print("Response is not JSON:", response.text)  # Handle non-JSON response
        else:
            print(f"Error: Received status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Function to send SMS notification with accumulated coordinates
def send_alert(accumulated_coords):
    try:
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

# Function to filter out nearby potholes (increase threshold to 5 meters)
def filter_nearby_potholes(coords, threshold=5):
    filtered_coords = []
    for coord in coords:
        if coord not in sent_coords and all(geodesic(coord, c).meters > threshold for c in filtered_coords):
            filtered_coords.append(coord)
            sent_coords.add(coord)  # Add to sent coordinates to avoid repeating in future alerts
    return filtered_coords

# Function to convert GPS degrees, minutes, seconds to decimal format
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction == 'S' or direction == 'W':
        decimal *= -1
    return decimal

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
        # Regex to extract degrees, minutes, seconds, and direction
        lat_match = re.match(r'(\d+) deg (\d+)\x27 (\d+\.\d+)" ([N|S])', gps_data['Latitude'])
        lon_match = re.match(r'(\d+) deg (\d+)\x27 (\d+\.\d+)" ([E|W])', gps_data['Longitude'])

        if lat_match and lon_match:
            lat_degrees, lat_minutes, lat_seconds, lat_direction = lat_match.groups()
            lon_degrees, lon_minutes, lon_seconds, lon_direction = lon_match.groups()

            # Convert to decimal format
            latitude = dms_to_decimal(lat_degrees, lat_minutes, lat_seconds, lat_direction)
            longitude = dms_to_decimal(lon_degrees, lon_minutes, lon_seconds, lon_direction)
            return latitude, longitude
    return None

# Load the CSV file containing GPS data (previously recorded pothole locations)
csv_file_path = '/Users/shreeshnadgouda/Desktop/Pothole_Detection/cleaned_lat_long_data.csv'
data = pd.read_csv(csv_file_path)

# Strip extra spaces and clean columns
data.columns = data.columns.str.strip()

# Extract coordinates where potholes are detected (PCI == 7)
coordinates = data[data['PCI'] == 7][['Latitude', 'Longitude']].values

# Load the YOLO model
model = YOLO("/Users/shreeshnadgouda/Desktop/Pothole_Detection/best-3.pt")
class_names = model.names

# Initialize PyQt application for file selection
app = QApplication([])
file_path, _ = QFileDialog.getOpenFileName(None, "Select Image or Video", "", "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi *.mov)")
app.exit()

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Initialize variables for accumulating detections
accumulated_coords = []
frame_count = 0
notification_interval = 25  # Send notification every 25 frames

# Initial GPS coordinates for the starting point of the video (use first point from metadata)
gps_metadata = extract_gps_from_video(file_path)
if gps_metadata:
    last_known_gps = gps_metadata  # Use extracted GPS coordinates from video metadata
else:
    last_known_gps = coordinates[0]  # Default to CSV if no GPS metadata is found

# Function to check if detected pothole is within proximity to known pothole locations
def is_within_proximity(lat, lon, proximity_threshold=2):
    for coord in coordinates:
        dist = geodesic((lat, lon), (coord[0], coord[1])).meters
        if dist <= proximity_threshold:
            return True
    return False

# Function to process detections and accumulate coordinates
def process_detections(results, latitude, longitude, img):
    pothole_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0 and box.conf[0] > 0.3:  # Assuming class 0 is pothole
                pothole_detected = True
                x, y, w, h = box.xywh[0]
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)
                cv2.putText(img, "Pothole", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # If a pothole is detected, accumulate the coordinates if within proximity
    if pothole_detected and is_within_proximity(latitude, longitude):
        print(f"Pothole detected at Latitude: {latitude}, Longitude: {longitude}")
        accumulated_coords.append((latitude, longitude))
        # Send pothole data to Flask server
        send_pothole_data(latitude, longitude)

# Process image or video
if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    img = cv2.imread(file_path)
    latitude, longitude = last_known_gps  # Start from initial GPS
    results = model.predict(img)
    process_detections(results, latitude, longitude, img)
    cv2.imshow('Pothole Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
    cap = cv2.VideoCapture(file_path)
    frame_index = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % 1 == 0:  # Process every 3rd frame
            # Map current frame to GPS coordinates from CSV
            latitude, longitude = coordinates[frame_index % len(coordinates)]
            img = cv2.resize(img, (1020, 500))
            results = model.predict(img)
            process_detections(results, latitude, longitude, img)
            cv2.imshow('Pothole Detection', img)

            # After every 25 frames, send a cumulative alert with filtered coordinates
            frame_count += 1
            if frame_count >= notification_interval:
                if accumulated_coords:
                    # Filter coordinates to remove duplicates within 5 meters and check they haven't been sent already
                    filtered_coords = filter_nearby_potholes(accumulated_coords)
                    if filtered_coords:  # Only send if there are new coordinates
                        send_alert(filtered_coords)
                    accumulated_coords = []  # Reset the accumulated coordinates
                frame_count = 0

            # Control playback speed and allow quitting with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unsupported file format. Please upload a valid image or video.")
