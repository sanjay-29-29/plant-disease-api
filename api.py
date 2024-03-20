# Description: This file contains the code for the FastAPI server which serves as the backend for the plant disease detection and chatbot application.
# keywords = ['agriculture', 'plants', 'crop', 'farm', 'soil', 'fertilizer', 'pesticide', 'harvest', 'seed', 'irrigation']
from flask import Flask, request
from flask_cors import CORS
from pyngrok import ngrok
import torch
import io
import re
from PIL import Image
import torch.nn as nn
import threading
from threading import Event
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(-1)

app = Flask(__name__)
CORS(app) 

history = None
llm_model = None
tokenizer = None
    
def extract_text_from_multipart(query: str):
    pattern = r'------WebKitFormBoundary.*\r\nContent-Disposition: form-data; name="query"\r\n\r\n(.*)\r\n------WebKitFormBoundary'
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
    op_text = predict_image(resnet_model, "image.jpg")
    op_text = op_text.lower()
    if('healthy' in op_text):
        response = "The plant is healthy"
    else:
        query = 'give me prevention and fertilizers to use for' + op_text + 'in a detailed manner'
        response, history = llm_model.chat(tokenizer, query=query, history=history)
        print(response)
        history = history[-3:]
        return {"response": response}

@app.post("/text_query")
async def plant_image(query: str = Body(...)):
    global history, llm_model, tokenizer
    query = extract_text_from_multipart(query)
    print(query)
    response, history = llm_model.chat(tokenizer, query, history=history)
    print(response)
    history = history[-3:]
    return {"response": response}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
    ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
    public_url = ngrok.connect(8000)
    print(f"Public URL: {public_url}")