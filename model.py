import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_model():
    # Training data — mood, sleep, screen time se burnout predict karna
    np.random.seed(42)
    n = 500

    sleep = np.random.uniform(3, 10, n)
    screen = np.random.uniform(1, 14, n)
    mood = np.random.uniform(1, 10, n)

    # Burnout score calculate karo (0-100)
    burnout = (
        (10 - sleep) * 5 +
        screen * 4 +
        (10 - mood) * 5
    )
    burnout = np.clip(burnout, 0, 100)

    # High burnout = 1, Low = 0
    labels = (burnout > 60).astype(int)

    X = np.column_stack([sleep, screen, mood])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, labels)

    # Model save karo
    with open('burnout_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model trained and saved!")
    return model, scaler

def load_model():
    if not os.path.exists('burnout_model.pkl'):
        return train_model()
    with open('burnout_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_burnout(sleep_hours, screen_time, mood_score):
    model, scaler = load_model()
    X = np.array([[sleep_hours, screen_time, mood_score]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    score = int(probability * 100)
    
    if score < 30:
        risk = "Low"
        message = "Great! You are doing well. Keep it up!"
    elif score < 60:
        risk = "Medium"
        message = "Be careful. Take breaks and sleep well."
    else:
        risk = "High"
        message = "High burnout risk! Please rest and take care."
    
    return {
        "burnout_score": score,
        "risk_level": risk,
        "message": message,
        "prediction": int(prediction)
    }