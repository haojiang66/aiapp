import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Update path if needed

# Open the webcam (0 for built-in webcam, 1 for external)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv8 object detection
    results = model(frame)

    # Draw the detection results
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("Fish Detection", annotated_frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
