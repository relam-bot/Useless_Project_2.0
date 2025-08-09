import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

app = FastAPI(title="noozGPT Backend")

# CORS settings for frontend access
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # For production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


async def get_location_by_ip(ip: str):
    url = f"http://ip-api.com/json/{ip}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        logging.info(f"Location API response: {data}")
        if data['status'] == 'success':
            return {
                "city": data.get("city"),
                "region": data.get("regionName"),
                "country": data.get("country"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
            }
        else:
            return None


async def get_weather(lat: float, lon: float):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}&aqi=no"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        logging.info(f"Weather API status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logging.info(f"Weather API response: {data}")
            current = data.get("current", {})
            condition = current.get("condition", {}).get("text", "Unknown")
            temp_c = current.get("temp_c", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_kph = current.get("wind_kph", "N/A")

            return {
                "condition": condition,
                "temperature_c": temp_c,
                "humidity": humidity,
                "wind_kph": wind_kph,
            }
        else:
            return None


def get_time_context(timezone_str="Asia/Kolkata"):
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    hour = now.hour
    weekday = now.strftime("%A")

    if 7 <= hour <= 10:
        time_period = "morning rush hour"
    elif 16 <= hour <= 19:
        time_period = "evening rush hour"
    else:
        time_period = "off-peak hours"

    is_weekend = weekday in ["Saturday", "Sunday"]

    return {
        "hour": hour,
        "weekday": weekday,
        "time_period": time_period,
        "is_weekend": is_weekend,
    }


async def get_news(country_code="us"):
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"country={country_code}&"
        f"apiKey={NEWS_API_KEY}&"
        f"pageSize=5"
    )
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        logging.info(f"News API status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logging.info(f"News API response: {data}")
            articles = data.get("articles", [])
            return [
                {
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "url": article.get("url"),
                }
                for article in articles
            ]
        else:
            return None


async def get_public_transport_status(location):
    # MOCK implementation - replace with real API if available
    logging.info(f"Fetching public transport status for {location}")
    return {
        "status": "Normal service",
        "note": "No delays reported on main bus and metro lines."
    }


def generate_excuse_text(location, weather, time_info, news_headlines, transport_status, role):
    prompt = f"""
You are a witty personal assistant. Generate a short, casual, believable excuse for being late as a {role} based on this info:

Location: {location['city']}, {location['region']}, {location['country']}
Weather: {weather['condition']}, Temp: {weather['temperature_c']}°C, Humidity: {weather['humidity']}%, Wind: {weather['wind_kph']} kph
Time: {time_info['weekday']}, {time_info['time_period']} (Hour: {time_info['hour']})
Public transport status: {transport_status['status']}. Note: {transport_status['note']}
Recent news headlines:
"""

    for i, article in enumerate(news_headlines, 1):
        prompt += f"{i}. {article['title']} (Source: {article['source']})\n"

    prompt += """
Make the excuse for being late based on ur role,also makei it based on news,time,wheather if it is valid and be human.
Excuse:
"""

    logging.info(f"Prompt sent to Gemini:\n{prompt}")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        logging.info(f"Gemini response: {response.text.strip()}")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return "Sorry, I couldn't come up with a good excuse right now."


@app.post("/generateExcuse")
async def generate_excuse(request: Request):
    data = await request.json()
    role = data.get("role", "someone")

    ip = request.client.host
    if ip.startswith("127.") or ip == "::1":
        ip = "8.8.8.8"

    logging.info(f"Request IP: {ip}")

    loc_info = await get_location_by_ip(ip)
    logging.info(f"Location info: {loc_info}")
    if not loc_info:
        return JSONResponse(content={"error": f"Could not determine location from IP: {ip}"})

    weather = await get_weather(loc_info["lat"], loc_info["lon"])
    logging.info(f"Weather info: {weather}")
    if not weather:
        return JSONResponse(content={"error": "Could not fetch weather data"})

    time_info = get_time_context()
    logging.info(f"Time info: {time_info}")

    news = await get_news()
    logging.info(f"News headlines: {news}")
    if news is None:
        news = []

    transport_status = await get_public_transport_status(loc_info)
    logging.info(f"Transport status: {transport_status}")

    excuse = generate_excuse_text(loc_info, weather, time_info, news, transport_status, role)
    logging.info(f"Generated excuse: {excuse}")

    return {
        "ip_used": ip,
        "location": loc_info,
        "weather": weather,
        "time_info": time_info,
        "news_headlines": news,
        "public_transport_status": transport_status,
        "excuse": excuse,
    }
