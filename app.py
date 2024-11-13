from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variable to store detected pothole coordinates
pothole_coords = []

# Route for the main map page
@app.route('/')
def index():
    # Simply render the HTML template, the map will be dynamically created with JavaScript
    return render_template('pothole_map.html')

# Route to update pothole coordinates in real-time
@app.route('/update_potholes', methods=['POST'])
def update_potholes():
    global pothole_coords
    # Append the new detected pothole coordinates from the POST request
    new_coords = (request.json['lat'], request.json['lon'])
    pothole_coords.append(new_coords) 
    return jsonify(success=True)

# Route to serve real-time pothole updates
@app.route('/get_pothole_updates', methods=['GET'])
def get_pothole_updates():
    return jsonify({'potholes': pothole_coords})

if __name__ == '__main__':
    app.run(debug=True)
