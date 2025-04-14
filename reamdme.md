https://www.no-how.com/knowhow/SHOTA250414071919z3x83r


# セル①：Notebook冒頭に必ず置く
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
