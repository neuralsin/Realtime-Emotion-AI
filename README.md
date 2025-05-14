# Realtime-Emotion-AI
Real-time face emotion detection using YOLOv8 and DeepFace

## 🔍 How It Works

- Uses YOLOv8 to detect faces from live webcam feed
- Crops the face region and sends it to DeepFace
- DeepFace analyzes emotions like happy, sad, angry, etc.
- Outputs live webcam feed with bounding boxes and emotion labels

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt

