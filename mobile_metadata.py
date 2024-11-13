import os
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client  # Twilio client for sending SMS
from geopy.distance import geodesic  # For calculating distance between coordinates
from PyQt5.QtWidgets import QApplication, QFileDialog

# Initialize Twilio client (replace with your actual credentials)
account_sid = 'ACcd5104988b78c92c52553d526198c15d'
auth_token = '70d1d854cd2aa7c23dc29db77ab63b0d'
twilio_phone_number = '+1 814 205 0137'
recipient_phone_number = '+917710879977'
client = Client(account_sid, auth_token)

# Set to keep track of already sent coordinates
sent_coords = set()

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

# Function to filter out nearby potholes (within a threshold distance of 20 meters)
def filter_nearby_potholes(coords, threshold=20):
    filtered_coords = []
    for coord in coords:
        # Only add coordinates that are not already sent and are unique within the distance threshold
        if coord not in sent_coords and all(geodesic(coord, c).meters > threshold for c in filtered_coords):
            filtered_coords.append(coord)
            sent_coords.add(coord)  # Add to sent coordinates to avoid repeating in future alerts
    return filtered_coords

# Function to extract GPS metadata from video using exiftool
def extract_gps_from_video(video_file_path):
    process = subprocess.Popen(['exiftool', video_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    
    metadata = out.decode('utf-8').split('\n')
    gps_data = {}
    
    for line in metadata:
        if 'GPS Latitude' in line:
            gps_data['Latitude'] = line.split(': ')[1]
        if 'GPS Longitude' in line:
            gps_data['Longitude'] = line.split(': ')[1]

    if 'Latitude' in gps_data and 'Longitude' in gps_data:
        return gps_data
    else:
        return None

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

# Initialize variables for accumulating detections
accumulated_coords = []
frame_count = 0
notification_interval = 20  # Send notification every 20 frames

# Extract GPS coordinates from the video (for .MOV, .MP4, etc.)
gps_data = extract_gps_from_video(file_path)
if gps_data:
    print(f"Extracted GPS Coordinates: Latitude {gps_data['Latitude']}, Longitude {gps_data['Longitude']}")
else:
    print("No GPS data found in the video.")
    exit()

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

    # If a pothole is detected, accumulate the coordinates
    if pothole_detected:
        accumulated_coords.append((latitude, longitude))

# Process image or video
if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    img = cv2.imread(file_path)
    latitude, longitude = gps_data['Latitude'], gps_data['Longitude']
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
        if frame_index % 3 == 0:  # Process every 3rd frame
            latitude, longitude = gps_data['Latitude'], gps_data['Longitude']
            img = cv2.resize(img, (1020, 500))
            results = model.predict(img)
            process_detections(results, latitude, longitude, img)
            cv2.imshow('Pothole Detection', img)

            # After every 5 frames, send a cumulative alert with filtered coordinates
            frame_count += 1
            if frame_count >= notification_interval:
                if accumulated_coords:
                    # Filter coordinates to remove duplicates within 20 meters and check they haven't been sent already
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
