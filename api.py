import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uvicorn
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import threading
import re
import ngrok
from PIL import Image
import utils
from googletrans import Translator

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
ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="glowing-polite-porpoise.ngrok-free.app")
translator = Translator()

def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'  # Adjusted pattern
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Could not find query text within multipart data")

def tamil_translate(text):
    global translator
    tamil_text = translator.translate(text, src='en', dest='ta').text
    return tamil_text

@app.post("/english_image_query")
async def plant_image(image: UploadFile = File(...)):
    global history, alexnet_model, llm_model, tokenizer
    image_content = await image.read()
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
            img.save("image.jpg")
    except Exception as e:
        print(e)
        return {"error": "Error in image processing"}
    
    #query = extract_text_from_multipart(query)
    op_text = utils.predict_image(alexnet_model, "image.jpg")
    op_text = op_text.lower()
    if('healthy' in op_text):
        return {'response': "The plant is healthy. If you have any other queries, feel free to ask."}
    else:
        query = 'give me prevention and fertilizers to use for' + op_text + 'in a detailed manner'
        response, history = llm_model.chat(tokenizer, query=query, history=history)
        history = history[-3:]
        response = 'The plant is suffering from ' + op_text + '. ' + response
        print(response) 
        return {"response": response}

@app.post("/english_text_query")
async def plant_image(query: str = Body(...)):
    global history, llm_model, tokenizer
    query = extract_text_from_multipart(query)
    print(query)
    response, history = llm_model.chat(tokenizer, query, history=history)
    history = history[-3:]
    print(response)
    return {"response": response}
    
@app.post("/tamil_image_query")
async def plant_image(image: UploadFile = File(...)):
    global history, alexnet_model, llm_model, tokenizer
    image_content = await image.read()
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
            img.save("image.jpg")
    except Exception as e:
        print(e)
        return {"error": "Error in image processing"}
    
    #query = extract_text_from_multipart(query)
    op_text = utils.predict_image(alexnet_model, "image.jpg")
    op_text = op_text.lower()
    if('healthy' in op_text):
        return {'response': "கொடுக்கப்பட்ட இலை ஆரோக்கியமானது. உங்களுக்கு வேறு ஏதேனும் கேள்விகள் இருந்தால், தயங்காமல் கேளுங்கள்...."}
    else:
        query = 'give me prevention and fertilizers to use for' + op_text + 'in a detailed manner'
        response, history = llm_model.chat(tokenizer, query=query, history=history)
        history = history[-3:]
        response = 'The plant is suffering from ' + op_text + '. ' + response
        response = str(tamil_translate(response))
        print(response) 
        return {"response": response}

@app.post("/tamil_text_query")
async def plant_image(query: str = Body(...)):
    global history, llm_model, tokenizer, translate
    query = extract_text_from_multipart(query)
    detected = translator.detect(query)
    if detected.lang == 'ta':
        query = translator.translate(query, dest='en').text
    print(query)
    response, history = llm_model.chat(tokenizer, query, history=history)
    history = history[-3:]
    response = str(tamil_translate(response))
    print(response)
    return {"response": response}
