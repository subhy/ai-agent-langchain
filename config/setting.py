import os
from dotenv import load_dotenv

load_dotenv()

#------LLM------------------------
LLM_PROVIDER  = os.getenv("LLM_PROVIDER","openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY","")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

#---------Agent-------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS",10))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))

#----------------API---------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
