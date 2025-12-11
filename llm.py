from config import Config
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
def initialize_llm():
    """Initialize and return the language model."""
    return ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)