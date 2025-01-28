from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import time
import torch
import ollama

# Загружаем API-токен из .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

# URL API для модели Mistral-7B


response = ollama.chat(model="llama2", messages=[{"role": "user", "content": "2+2 = ?"}])
print(response["message"]["content"])



# prompt = 'Hello, how are you?'

# url = "http://localhost:11434/api/generate"
# data = {"model": "llama2", "prompt": "Привет!"}

# response = requests.post(url, json=data)
# print(response.json())

# response = ollama.chat(model='llama2',messages=[{'role':'user','content': promt}])

# print(response['message']['content'])










# client = OpenAI(api_key=key_api)

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )

# print(completion.choices[0].message)