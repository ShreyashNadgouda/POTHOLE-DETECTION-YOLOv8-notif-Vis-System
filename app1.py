from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load CSV data containing pothole locations
csv_path = '/Users/shreeshnadgouda/Desktop/Pothole_Detection/cleaned_lat_long_data.csv'
data = pd.read_csv(csv_path)
EARTH_RADIUS_KM = 6371
pothole_coords = []  # Global variable to store detected pothole coordinates
fixed_video_location = None

# Calculate distance between two points using the Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c

# Find potholes within a specified radius of the video location
def get_potholes_near_location(video_lat, video_long, radius_km=0.1):
    if video_lat is None or video_long is None:
        logging.warning("Invalid video location coordinates received.")
        return []

    # Filter potholes within the specified radius
    pothole_data = data[(data['PCI'] == 7) & data['Latitude'].notna() & data['Longitude'].notna()]
    nearby_potholes = pothole_data[
        pothole_data.apply(
            lambda row: haversine_distance(video_lat, video_long, row['Latitude'], row['Longitude']) <= radius_km,
            axis=1
        )
    ]
    return nearby_potholes[['Latitude', 'Longitude']].to_dict(orient='records')

@app.route('/')
def index():
    return render_template('pothole_map.html')


@app.route('/update_potholes', methods=['POST'])
def update_potholes():
    global pothole_coords, fixed_video_location
    data = request.json
    logging.info(f"Received data payload: {data}")

    # Get GPS coordinates from detection script
    video_lat = data.get('lat')
    video_long = data.get('lon')
    detection_active = data.get('detection_active', False)

    # Validate coordinates
    if video_lat is None or video_long is None:
        return jsonify(success=False, error="Invalid coordinates"), 400

    # Set fixed location only once, based on initial detection
    if detection_active:
        if fixed_video_location is None:
            fixed_video_location = (video_lat, video_long)
            logging.info(f"Set fixed video location to: {fixed_video_location}")
        
        # Clear and update pothole coordinates based on proximity to fixed location
        pothole_coords.clear()
        nearby_potholes = get_potholes_near_location(fixed_video_location[0], fixed_video_location[1], radius_km=0.1)
        pothole_coords.extend([(p['Latitude'], p['Longitude']) for p in nearby_potholes])
        logging.info(f"Updated pothole coordinates: {pothole_coords}")

    return jsonify(success=True, nearby_potholes=pothole_coords)

@app.route('/get_pothole_updates', methods=['GET'])
def get_pothole_updates():
    # Return available pothole coordinates for map markers
    if not pothole_coords:
        logging.info("No pothole coordinates available.")
        return jsonify({"potholes": []})
    
    return jsonify({"potholes": [{"lat": coord[0], "lon": coord[1]} for coord in pothole_coords]})

if __name__ == '__main__':
    app.run(debug=True)
