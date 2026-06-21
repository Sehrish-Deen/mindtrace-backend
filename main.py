from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_burnout, train_model
import numpy as np
import base64
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
        from fer import FER
        import numpy as np
        import cv2
        from PIL import Image, ImageOps
        import io

        # Base64 decode karo
        image_data = base64.b64decode(data.image_base64)

        # PIL se image kholo aur EXIF orientation fix karo
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image = ImageOps.exif_transpose(pil_image)  # rotation fix
        pil_image = pil_image.convert('RGB')

        # PIL image ko OpenCV (BGR) format mein convert karo
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if img is None:
            return {
                "success": False,
                "detected_mood": "😐 Neutral",
                "mood_score": 5,
                "suggestion": "Could not read image. Please try again!"
            }

        # FER se emotion detect karo
        detector = FER(mtcnn=False)
        result = detector.detect_emotions(img)

        if not result:
            return {
                "success": False,
                "detected_mood": "😐 Neutral",
                "mood_score": 5,
                "suggestion": "No face detected. Please look at camera!"
            }

        emotions = result[0]['emotions']
        dominant_emotion = max(emotions, key=emotions.get)

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

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "detected_mood": "😐 Neutral",
            "mood_score": 5,
            "suggestion": "Could not detect emotion. Please try again!"
        }