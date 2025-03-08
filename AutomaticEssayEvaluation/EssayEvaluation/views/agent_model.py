from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.agent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = 'mistral:latest'
OLLAMA_SERVE_ENDPOINT = 'http://localhost:11434/v1'

TOGETHER_API_KEY = os.getenv('API_KEY')
MODEL_ID_TOGETHER = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOGETHER_SERVE_ENDPOINT = "https://api.together.xyz/v1"


model_ollama = OpenAIModel(
    model_name=MODEL_ID,
    base_url=OLLAMA_SERVE_ENDPOINT,
)
model_together = OpenAIModel(
    model_name=MODEL_ID_TOGETHER,
    base_url=TOGETHER_SERVE_ENDPOINT,
    api_key=TOGETHER_API_KEY
)


agent = Agent(
    # model= model_together
    model= model_ollama
)
