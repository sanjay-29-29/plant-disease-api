import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import utils
import uvicorn
import os
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import threading
import re
import ngrok
from PIL import Image
original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

if original_cuda_visible_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
else:
    del os.environ['CUDA_VISIBLE_DEVICES']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

alexnet_model = utils.load_model_alexnet()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True) 
llm_model = AutoModelForCausalLM.from_pretrained("sanjay-29-29/GreenAI", trust_remote_code=True, device_map='auto') 
history = None

def run_server():
    ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
    listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="glowing-polite-porpoise.ngrok-free.app")
    uvicorn.run("api:app", host="127.0.0.1", port=8000)

def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'  # Adjusted pattern
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Could not find query text within multipart data")
    

@app.post("/image_query")
async def plant_image(query: str = Body(...), image: UploadFile = File(...)):
    global history, resnet_model, llm_model, tokenizer
    image_content = await image.read()
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
            img.save("image.jpg")
    except Exception as e:
        print(e)
        return {"error": "Error in image processing"}
    
    query = extract_text_from_multipart(query)
    op_text = utils.predict_image(alexnet_model, "image.jpg")
    op_text = op_text.lower()
    if('healthy' in op_text):
        response = "The plant is healthy"
    else:
        query = 'give me prevention and fertilizers to use for' + op_text + 'in a detailed manner'
        response, history = llm_model.chat(tokenizer, query=query, history=history)
        history = history[-3:]
        return {"response": response}

@app.post("/text_query")
async def plant_image(query: str = Body(...)):
    global history, llm_model, tokenizer
    query = extract_text_from_multipart(query)
    print(query)
    response, history = llm_model.chat(tokenizer, query, history=history)
    history = history[-3:]
    return {"response": response}

if __name__ == "__main__":
    resnet_model = utils.load_model_weights(utils.ResNet9(3,38), 'plant-disease-model-complete.pth')
    threading.Thread(target=run_server).start()