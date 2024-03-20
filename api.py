# Description: This file contains the code for the FastAPI server which serves as the backend for the plant disease detection and chatbot application.
import torch
import io
import re
import ngrok
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from restnet import ResNet9, predict_image
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

torch.seed(-1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True) 
llm_model = AutoModelForCausalLM.from_pretrained("sanjay-29-29/GreenAI", trust_remote_code=True, device_map='auto')
resnet_model = ResNet9(3, 38)
resnet_model = torch.load('plant-disease-model-complete.pth', map_location=torch.device('cpu')) 
history = None
ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="glowing-polite-porpoise.ngrok-free.app")

keywords = ['agriculture', 'plants', 'crop', 'farm', 'soil', 'fertilizer', 'pesticide', 'harvest', 'seed', 'irrigation']

def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Could not find query text within multipart data")

@app.post("/image_query")
async def plant_image(query: str = Body(...), image: UploadFile = File(...)):
    global history
    image_content = await image.read()
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
            img.save("image.jpg")
    except Exception as e:
        print(e)
        return {"error": "Error in image processing"}
    
    query = extract_text_from_multipart(query)
    op_text = predict_image(resnet_model, "image.jpg")
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
    global history
    query = extract_text_from_multipart(query)
    print(query)
    response, history = llm_model.chat(tokenizer, query, history=history)
    history = history[-3:]
    return {"response": response}
