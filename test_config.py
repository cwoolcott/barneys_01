from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if key:
    print("✅ API key loaded successfully")
else:
    print("❌ API key not found — check your .env file")
