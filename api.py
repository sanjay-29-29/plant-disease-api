from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uvicorn
import os
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import threading
import re
from pyngrok import ngrok
from PIL import Image

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

def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'  # Adjusted pattern
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Could not find query text within multipart data")
    

@app.post("/image_query")
async def plant_image(query: str = Body(...), image: UploadFile = File(...)):
    image_content = await image.read()
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
            img.save("image.jpg")
    except Exception as e:
        print(e)
        return {"error": "Error in image processing"}
    
    #text = extract_text_from_multipart(query)

    print(query)

    '''query = tokenizer.from_list_format([
    {'image': 'image.jpg'},
    {'text': text},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    return {"response": response}'''

@app.post("/text_query")
async def plant_image(query: str = Body(...)):
    query = extract_text_from_multipart(query)
    print(query)
    response, history = model.chat(tokenizer, query, history=None)
    return {"response": response}
