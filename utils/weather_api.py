
import requests

def get_weather(api_key: str, city: str, units: str = 'metric'):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units={units}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    main = data.get('main', {})
    return {
        'temperature': main.get('temp'),
        'humidity': main.get('humidity'),
    }
