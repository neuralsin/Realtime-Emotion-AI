import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import mediapipe as mp

# Load models
yolo_model = YOLO("yolov8n.pt")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLO detection
    results = yolo_model(frame, verbose=False)

    # Go through detections
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = map(int, result[:6])
        if int(cls) == 0:  # person class
            face_region = frame[y1:y2, x1:x2]

            if face_region.size > 0:
                try:
                    # DeepFace emotion analysis
                    analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
                    confidence = analysis[0]['emotion'][dominant_emotion]

                    label = f"{dominant_emotion}: {confidence:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                except Exception as e:
                    print(f"DeepFace error: {e}")

    # Face mesh 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = mp_face_mesh.process(rgb)

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )

    # result
    cv2.imshow("🔥 Emotion + Skeleton Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
