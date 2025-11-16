from google import genai
from dotenv import load_dotenv
import os
import sys

# Load .env from the current working directory
# (place your .env next to this script or set an absolute path)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found. Create a .env file with:\nGEMINI_API_KEY=AIza...yourkey")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# List all models
models = client.models.list()

print("Available Gemini Models:\n")
for m in models:
    print(m.name)
