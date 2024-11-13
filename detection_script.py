import requests

def send_pothole_data(lat, lon):
    url = 'http://127.0.0.1:5000/update_potholes'
    data = {'lat': lat, 'lon': lon, 'detection_active': True}  # Send detection_active as true
    try:
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            print("Data sent successfully:", response.json())
        else:
            print("Error:", response.status_code)
    except requests.RequestException as e:
        print("Request failed:", e)

# Only call `send_pothole_data` when detection is actually active
