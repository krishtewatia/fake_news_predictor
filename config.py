import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
