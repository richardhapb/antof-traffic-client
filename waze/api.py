import requests
from config import WAZE_API_URL


class WazeAPI:
    def __init__(self):
        self.url = WAZE_API_URL

    def get_data(self):
        response = requests.get(self.url, timeout=10)
        data = response.json()
        return data
