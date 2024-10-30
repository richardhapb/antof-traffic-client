import requests


class WazeAPI:
    def __init__(self):
        self.url = "https://www.waze.com/row-partnerhub-api/partners/18532407453/waze-feeds/d44195e2-2952-4b2f-8539-af8e85b661c5?format=1"

    def get_data(self):
        response = requests.get(self.url, timeout=10)
        data = response.json()
        return data
