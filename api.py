import os


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_model_alexnet():

    classifier = Sequential()

    classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 3
    classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    classifier.add(Flatten())

    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 1000, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 38, activation = 'softmax'))

    classifier.load_weights('/kaggle/input/alexnet/tensorflow2/v1.0/1/AlexNetModel.hdf5')

    return classifier

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
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite',
    'Tomato Target_Spot',
    'Tomato_Yellow Leaf Curl Virus',
    'Tomato mosaic virus',
    'Tomato healthy'
    ]
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = tf.nn.softmax(pred, axis=1).numpy()
    pred = pred[0]
    pred = pred.tolist()
    pred = {diseases[i]: pred[i] for i in range(len(diseases))}
    class_max_prob = max(pred, key=pred.get)
    
    return class_max_prob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

alexnet_model = load_model_alexnet()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True) 
llm_model = AutoModelForCausalLM.from_pretrained("sanjay-29-29/GreenAI", trust_remote_code=True, device_map='auto') 
history = None
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
    op_text = predict_image(alexnet_model, "image.jpg")
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
