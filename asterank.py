import requests
import pandas as pd

# API endpoint and parameters
api_endpoint = "https://api.openweathermap.org/data/2.5/weather"
api_endpoint2 = 'http://asterank.com/api/asterank?query={"e":{"$lt":0.1},"i":{"$lt":4},"a":{"$lt":1.5}}&limit=1'


# Make API request
response = requests.get(api_endpoint2)
print(response)

data = response.json()
print(data)
'''
# Extract relevant information
temperature = data["main"]["temp"]
humidity = data["main"]["humidity"]
wind_speed = data["wind"]["speed"]

# Create a Pandas DataFrame
weather_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed]
})
'''