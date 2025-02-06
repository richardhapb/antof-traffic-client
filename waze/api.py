import requests
from config import WAZE_API_URL


class WazeAPI:
    def __init__(self):
        self.url = WAZE_API_URL

    def get_data(self):
        if not self.url:
            raise ValueError("URL de la API de Waze no configurada")
        response = requests.get(self.url, timeout=10)
        data = response.json()
        return data
