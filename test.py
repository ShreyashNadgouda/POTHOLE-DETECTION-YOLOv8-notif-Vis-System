import cv2
import pandas as pd
from ultralytics import YOLO
from twilio.rest import Client  # Twilio client for sending SMS
from geopy.distance import geodesic  # For calculating distance between coordinates
from PyQt5.QtWidgets import QApplication, QFileDialog
import subprocess
import re
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
    response = requests.post(url, json=data)
    print(response.json())  # Log server response

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

# Function to filter potholes within a certain distance from initial coordinates
def filter_potholes_within_boundary(pothole_coords, initial_coord, boundary_radius=100):
    filtered_coords = []
    for coord in pothole_coords:
        dist = geodesic(initial_coord, coord).meters
        if dist <= boundary_radius:
            filtered_coords.append(coord)
    return filtered_coords

# Function to check if detected pothole is within proximity to known pothole locations
def is_within_proximity(lat, lon, proximity_threshold=2):
    for coord in coordinates:
        dist = geodesic((lat, lon), (coord[0], coord[1])).meters
        if dist <= proximity_threshold:
            return True
    return False

# Function to process detections and accumulate coordinates
def process_detections(results, latitude, longitude, img, initial_coord, boundary_radius=100):
    pothole_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0 and box.conf[0] > 0.3:  # Assuming class 0 is pothole
                pothole_detected = True
                x, y, w, h = box.xywh[0]
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + w / 2)), (0, 0, 255), 2)
                cv2.putText(img, "Pothole", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if pothole_detected:
        if geodesic((latitude, longitude), initial_coord).meters <= boundary_radius:
            accumulated_coords.append((latitude, longitude))
            send_pothole_data(latitude, longitude)

# Load the CSV file containing GPS data (previously recorded pothole locations)
csv_file_path = '/Users/shreeshnadgouda/Desktop/Pothole_Detection/cleaned_lat_long_data.csv'
data = pd.read_csv(csv_file_path)

data.columns = data.columns.str.strip()

coordinates = data[data['PCI'] == 7][['Latitude', 'Longitude']].values

# Load the YOLO model
model = YOLO("/Users/shreeshnadgouda/Downloads/best-2.pt")
class_names = model.names

# Initialize PyQt application for file selection
app = QApplication([])
file_path, _ = QFileDialog.getOpenFileName(None, "Select Image or Video", "", "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi *.mov)")
app.exit()

if not file_path:
    print("No file selected. Exiting...")
    exit()

accumulated_coords = []
frame_count = 0
notification_interval = 25

gps_metadata = extract_gps_from_video(file_path)
if gps_metadata:
    last_known_gps = gps_metadata
else:
    last_known_gps = coordinates[0]

# Corrected video processing loop
if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
    cap = cv2.VideoCapture(file_path)
    frame_index = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break  # Exit the loop when the video ends

        # Process every frame, or every few frames as necessary
        frame_index += 1
        if frame_index % 3 == 0:  # Skip some frames to optimize processing if needed
            latitude, longitude = last_known_gps  # Using metadata or CSV data
            img = cv2.resize(img, (1020, 500))
            results = model.predict(img)
            process_detections(results, latitude, longitude, img, last_known_gps, boundary_radius=100)
            cv2.imshow('Pothole Detection', img)

            # Count frames for notification intervals
            frame_count += 1
            if frame_count >= notification_interval:
                if accumulated_coords:
                    filtered_coords = filter_nearby_potholes(accumulated_coords)
                    if filtered_coords:
                        send_alert(filtered_coords)
                    accumulated_coords = []
                frame_count = 0

            # Allow quitting the video playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Handle image files separately if needed
elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    img = cv2.imread(file_path)
    latitude, longitude = last_known_gps
    results = model.predict(img)
    process_detections(results, latitude, longitude, img, last_known_gps, boundary_radius=100)
    cv2.imshow('Pothole Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Unsupported file format. Please upload a valid image or video.")