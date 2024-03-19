from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uvicorn
import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import threading
import re
from pyngrok import ngrok

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:tbd", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained("KissanAI/Dhenu-vision-lora-0.1", trust_remote_code=True, device_map='auto') 

ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
tunnel = ngrok.connect(8000)
print(f"Public URL: {tunnel.public_url}")
count = 0

def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'  # Adjusted pattern
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Could not find query text within multipart data")

@app.post("/text_query")
async def plant_image(query: str = Body(...)):
    print(query)
    global count
    if count == 0:
        response, history = model.chat(tokenizer, query, history=None)
    else:
        response, history = model.chat(tokenizer, query, history=history)
    count += 1
    
    return {"response": response}
