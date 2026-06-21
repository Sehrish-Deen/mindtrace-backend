from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_burnout, train_model
import numpy as np
import base64
import cv2
import tempfile
import os

app = FastAPI(title="MindTrace AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BurnoutRequest(BaseModel):
    sleep_hours: float
    screen_time: float
    mood_score: float

class ImageRequest(BaseModel):
    image_base64: str

@app.get("/")
def home():
    return {"message": "MindTrace AI Backend Running! ✅"}

@app.post("/predict")
def predict(data: BurnoutRequest):
    result = predict_burnout(
        sleep_hours=data.sleep_hours,
        screen_time=data.screen_time,
        mood_score=data.mood_score
    )
    return result

@app.get("/train")
def train():
    train_model()
    return {"message": "Model trained successfully! ✅"}

@app.post("/detect-emotion")
async def detect_emotion(data: ImageRequest):
    try:
        from deepface import DeepFace

        # Base64 image decode karo
        image_data = base64.b64decode(data.image_base64)
        
        # Temp file mein save karo
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        try:
            # DeepFace se emotion detect karo
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']

            # Mood score calculate karo
            emotion_scores = {
                'happy': 9,
                'surprise': 7,
                'neutral': 5,
                'sad': 3,
                'fear': 2,
                'angry': 2,
                'disgust': 1
            }

            mood_score = emotion_scores.get(dominant_emotion.lower(), 5)

            # Suggestions
            suggestions = {
                'happy': '😄 Great mood! Perfect time to tackle important tasks!',
                'surprise': '😮 You seem surprised! Take a moment to process things.',
                'neutral': '😐 Neutral mood. Take a short break and stretch!',
                'sad': '😢 You seem sad. Try deep breathing or talk to a friend.',
                'fear': '😨 You seem anxious. Try 4-7-8 breathing technique.',
                'angry': '😠 You seem stressed! Step away for 10 minutes.',
                'disgust': '🤢 Something bothering you? Take a break!'
            }

            emoji_map = {
                'happy': '😄 Happy',
                'surprise': '😮 Surprised',
                'neutral': '😐 Neutral',
                'sad': '😢 Sad',
                'fear': '😨 Anxious',
                'angry': '😠 Angry',
                'disgust': '🤢 Disgusted'
            }

            return {
                "success": True,
                "dominant_emotion": dominant_emotion,
                "detected_mood": emoji_map.get(dominant_emotion.lower(), '😐 Neutral'),
                "mood_score": mood_score,
                "emotion_probabilities": emotions,
                "suggestion": suggestions.get(dominant_emotion.lower(), 'Take a short break!')
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "dominant_emotion": "neutral",
            "detected_mood": "😐 Neutral",
            "mood_score": 5,
            "suggestion": "Could not detect emotion. Please try again!"
        }