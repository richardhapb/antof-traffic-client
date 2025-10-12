import os
from dotenv import load_dotenv

load_dotenv()

WAZE_API_URL = os.getenv("WAZE_API_URL")

SERVER_URL = os.getenv("SERVER_URL")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

