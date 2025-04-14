import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEYが設定されていません")
    return OpenAI(api_key=key)
