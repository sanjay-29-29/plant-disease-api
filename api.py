# Description: This file contains the code for the FastAPI server which serves as the backend for the plant disease detection and chatbot application.
# keywords = ['agriculture', 'plants', 'crop', 'farm', 'soil', 'fertilizer', 'pesticide', 'harvest', 'seed', 'irrigation']

import torch
import io
import re
import ngrok
from PIL import Image
import torch.nn as nn
import threading
from threading import Event
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

torch.seed()

app = FastAPI()

resnet_model = None
llm_model = None
history = None
tokenizer = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.on_event("startup")
def load_model():
    global llm_model, history, tokenizer
    llm_model = AutoModelForCausalLM.from_pretrained("sanjay-29-29/GreenAI", trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    history = None

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def create_model_resnet():
    model = ResNet9(3,38)
    model = torch.load('plant-disease-model-complete.pth', map_location=torch.device('cpu'))
    return model    
    
def predict_image(model, image_path):
    diseases = ['Apple scab',
    'Apple Black rot',
    'Apple Cedar_apple rust',
    'Apple healthy',
    'Blueberry healthy',
    'Cherry Powdery_mildew',
    'Cherry healthy',
    'Maize Cercospora leaf spot',
    'Maize Common rust',
    'Maize Northern Leaf Blight',
    'Maize healthy',
    'Grape Black rot',
    'Grape Esca Black Measles',
    'Grape Leaf blight Isariopsis Leaf Spot',
    'Grape healthy',
    'Orange Haunglongbing Citrus greening',
    'Peach Bacterial_spot',
    'Peach healthy',
    'Pepper bell Bacterial spot',
    'Pepper bell healthy',
    'Potato Early blight',
    'Potato Late_blight',
    'Potato healthy',
    'Raspberry healthy',
    'Soybean healthy',
    'Squash Powdery mildew',
    'Strawberry Leaf scorch',
    'Strawberry healthy',
    'Tomato Bacterial spot',
    'Tomato Early blight',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite',
    'Tomato Target_Spot',
    'Tomato_Yellow Leaf Curl Virus',
    'Tomato mosaic virus',
    'Tomato healthy'
    ]

    img = Image.open(image_path)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    pred = model(img)
    pred = F.softmax(pred, dim=1)
    pred = pred.detach().numpy()
    pred = pred[0]
    pred = pred.tolist()
    pred = {diseases[i]: pred[i] for i in range(len(diseases))}
    class_max_prob = max(pred, key=pred.get)
    
    return class_max_prob
    
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
    global resnet_model
    resnet_model = create_model_resnet()
    ngrok.set_auth_token("2dVBJw5G2bExzQ41keUUDtC0U8K_7zn55apnGM8YJ3RNsfznb")
    listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="glowing-polite-porpoise.ngrok-free.app")
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
