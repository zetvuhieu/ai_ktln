import requests
from datetime import datetime, timedelta, timezone

def fetch_weather_for_tomorrow():
    # Define the URL and parameters
    url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": "Buon Ma Thuot",
        "appid": "201bd92755af93923b3854e767deeed0",  # Replace with your OpenWeather API key
        "lang": "vi",
        "units": "metric",
        "cnt": 8
    }

    # Send the GET request to OpenWeather API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        
        # Get tomorrow's date
        tomorrow_date = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).date()
        
        # Initialize variables to store total temperature, humidity, and rainfall
        total_temp = 0
        total_humidity = 0
        total_rainfall = 0
        rain_count = 0
        entries_count = 0

        # Filter and gather temperature, humidity, and rainfall data only for tomorrow
        for entry in data['list']:
            entry_time = datetime.fromtimestamp(entry['dt'], tz=timezone.utc)
            if entry_time.date() == tomorrow_date:
                entries_count += 1
                # Add temperature and humidity
                total_temp += entry['main']['temp']
                total_humidity += entry['main']['humidity']
                
                # Check if rainfall data is available
                if 'rain' in entry:
                    total_rainfall += entry['rain'].get('3h', 0)  # Get rainfall for the 3-hour period
                    rain_count += 1

        # Calculate the averages
        if entries_count > 0:
            average_temp = total_temp / entries_count
            average_humidity = total_humidity / entries_count
            average_rainfall = total_rainfall / rain_count if rain_count > 0 else 0

            # Return the calculated averages as a tuple
            return average_temp, average_humidity, average_rainfall, tomorrow_date
        else:
            print("No data for tomorrow.")
            return None  # Return None if no data is available for tomorrow
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None  # Return None if API request fails
