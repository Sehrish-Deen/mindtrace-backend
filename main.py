from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_burnout, train_model

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