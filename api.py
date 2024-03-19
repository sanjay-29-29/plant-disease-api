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
    ngrok.forward("127.0.0.1:8000", authtoken_from_env=True)
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        print(f"Public URL: {tunnel.public_url}")
    uvicorn.run("api:app", host="127.0.0.1", port=8000)

@app.get("/text_query")
async def plant_image(query: str = Query(...)):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("KissanAI/Dhenu-vision-lora-0.1", trust_remote_code=True, device_map='auto')
    response, history = model.chat(tokenizer, query, history=history)
    return {"response": response}

if __name__ == "__main__":
    threading.Thread(target=run_server).start()