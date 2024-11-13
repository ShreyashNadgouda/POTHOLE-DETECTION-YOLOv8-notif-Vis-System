import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from PyQt5.QtWidgets import QApplication, QFileDialog

# Create a QApplication instance (required for using PyQt5 dialogs)
app = QApplication([])

# Let user choose between image or video upload
file_path, _ = QFileDialog.getOpenFileName(None, "Select Image or Video", "", 
                                           "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi)")

# Exit the QApplication after file selection
app.exit()

# Check if the file path is valid
if not file_path:
    print("No file selected. Exiting...")
    exit()

# Load the YOLO model
model = YOLO("/Users/shreeshnadgouda/Downloads/best-2.pt")
class_names = model.names

# Check if it's an image or video
if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
    # Load the image
    img = cv2.imread(file_path)
    h, w, _ = img.shape

    # Run YOLO model prediction on the image
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Bounding boxes for detected objects
        masks = r.masks  # Segmentation masks for detected objects

        if masks is not None:
            masks = masks.data.cpu()  # Convert masks to CPU if using GPU
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw contour around the detected object
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    # Display the class name above the bounding box
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the result
    cv2.imshow('Pothole Detection - Image', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

elif file_path.lower().endswith(('.mp4', '.avi')):
    # Process as video
    cap = cv2.VideoCapture(file_path)
    count = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

        # Resize the image for processing
        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape

        # Run YOLO model prediction on the frame
        results = model.predict(img)

        for r in results:
            boxes = r.boxes  # Bounding boxes for detected objects
            masks = r.masks  # Segmentation masks for detected objects

            if masks is not None:
                masks = masks.data.cpu()  # Convert masks to CPU if using GPU
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw contour around the detected object
                        cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                        # Display the class name above the bounding box
                        cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Pothole Detection - Video', img)

        # Use cv2.waitKey(1) to allow video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unsupported file format. Please upload a valid image or video.")
