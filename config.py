import os
from dotenv import load_dotenv

load_dotenv()

WAZE_API_URL = os.getenv("WAZE_API_URL")

DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_DATABASE", "waze"),
}
