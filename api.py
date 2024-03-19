from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uvicorn
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import threading
import ngrok

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

def run_server():
    ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
    listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="glowing-polite-porpoise.ngrok-free.app")
    uvicorn.run("api:app", host="127.0.0.1", port=8000)

def create_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("KissanAI/Dhenu-vision-lora-0.1", trust_remote_code=True, device_map='auto')
    return model, tokenizer

@app.get("/text_query")
async def plant_image(query: str = Query(...)):
    model, tokenizer = create_model()
    response, history = model.chat(tokenizer, query, history=None)
    return {"response": response}

if __name__ == "__main__":
    threading.Thread(target=run_server).start()