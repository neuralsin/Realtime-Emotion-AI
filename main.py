import cv2
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Replace with your path to YOLOv8 model

# Initialize the video capture (use webcam or video file)
cap = cv2.VideoCapture(0)  # Change '0' to your video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get YOLOv8 detections
    yolo_results = yolo_model(frame, verbose=False)
    
    # Loop through the detections
    for result in yolo_results[0]:
        # Only process face detections (class 0 is usually for people in YOLO)
        if result['class'] == 0:  # Face class (you can adjust this depending on YOLO version)
            x1, y1, x2, y2 = map(int, result['bbox'])  # Get bounding box coordinates
            face_region = frame[y1:y2, x1:x2]
            
            # Check if face region is non-empty before analyzing
            if face_region.size > 0:
                try:
                    # Analyze the face for emotion using DeepFace
                    analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                    
                    # Get the most confident emotion
                    dominant_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
                    confidence = analysis[0]['emotion'][dominant_emotion]

                    # Draw the bounding box and display the emotion near the face
                    label = f"{dominant_emotion}: {confidence:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # Display label

                except Exception as e:
                    print(f"Error analyzing face: {e}")
    
    # Show the frame with bounding boxes and emotions
    cv2.imshow("Real-time Face Emotion Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
