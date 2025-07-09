import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gemini API Key (required)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Input model
class HydrationInput(BaseModel):
    name: str
    age: int
    weight_kg: float
    gender: str
    city: str
    intake_today_ml: int
    activity_level: str  # sedentary, low, moderate, high

# Output model
class HydrationOutput(BaseModel):
    hydration_message: str
    predicted_goal_ml: int
    remaining_ml: int

# Dummy regression model
X_train = np.array([
    [30, 70, 25, 60, 2],
    [25, 60, 30, 70, 3],
    [40, 80, 20, 50, 1],
    [22, 65, 28, 65, 2],
    [35, 75, 22, 55, 1],
    [28, 68, 32, 75, 3],
    [50, 90, 18, 45, 0],
    [18, 55, 27, 62, 2]
])
y_train = np.array([2500, 3500, 2000, 2700, 2200, 3800, 1800, 2600])
model = LinearRegression()
model.fit(X_train, y_train)

def encode_activity_level(activity_level: str) -> int:
    mapping = {
        "sedentary": 0,
        "low": 1,
        "moderate": 2,
        "high": 3
    }
    return mapping.get(activity_level.lower(), 2)

def get_lat_lon(place):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={place}&count=1"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception("❌ Failed to fetch location coordinates")
    data = res.json()
    if "results" not in data or len(data["results"]) == 0:
        raise Exception("❌ Location not found")
    lat = data["results"][0]["latitude"]
    lon = data["results"][0]["longitude"]
    return lat, lon

def get_weather(city: str):
    try:
        lat, lon = get_lat_lon(city)
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m"
            f"&forecast_days=2&timezone=auto"
        )
        res = requests.get(url)
        if res.status_code != 200:
            raise Exception("❌ Failed to fetch weather data")

        data = res.json()
        temp = data["hourly"]["temperature_2m"][-24:]
        humidity = data["hourly"]["relative_humidity_2m"][-24:]

        avg_temp = round(sum(temp) / len(temp), 2)
        avg_humidity = round(sum(humidity) / len(humidity), 2)
        return avg_temp, avg_humidity

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Open-Meteo error: {str(e)}")

def predict_hydration_goal(age, weight, temp, humidity, activity_index):
    features = np.array([[age, weight, temp, humidity, activity_index]])
    return int(model.predict(features)[0])

def generate_prompt(name, age, weight, gender, activity_level, intake_ml, goal_ml, remaining_ml, temp, humidity, city):
    return f"""
You are a hydration assistant helping a user stay healthy and hydrated.

User Profile:
- Name: {name}
- Age: {age}
- Weight: {weight} kg
- Gender: {gender}
- Activity Level: {activity_level}

Today’s Context:
- City: {city}
- Temperature: {temp}°C
- Humidity: {humidity}%
- Water Intake So Far: {intake_ml} ml
- Predicted Daily Hydration Goal: {goal_ml} ml
- Remaining: {remaining_ml} ml

Write a short, friendly reminder message (under 25 words) encouraging the user to drink water based on this context.
"""

def ask_gemini(prompt: str) -> str:
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@app.post("/hydration-agent", response_model=HydrationOutput)
def hydration_agent(data: HydrationInput):
    # Step 1: Get weather data
    temp, humidity = get_weather(data.city)

    # Step 2: Encode activity level and predict hydration goal
    activity_index = encode_activity_level(data.activity_level)
    goal_ml = predict_hydration_goal(data.age, data.weight_kg, temp, humidity, activity_index)

    # Step 3: Calculate remaining
    remaining_ml = max(goal_ml - data.intake_today_ml, 0)

    # Step 4: Build prompt and ask Gemini
    prompt = generate_prompt(
        data.name,
        data.age,
        data.weight_kg,
        data.gender,
        data.activity_level,
        data.intake_today_ml,
        goal_ml,
        remaining_ml,
        temp,
        humidity,
        data.city
    )
    message = ask_gemini(prompt)

    return HydrationOutput(
        hydration_message=message.strip(),
        predicted_goal_ml=goal_ml,
        remaining_ml=remaining_ml
    )
