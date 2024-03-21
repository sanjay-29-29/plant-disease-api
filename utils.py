from PIL import Image
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

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

    classifier.load_weights('AlexNetModel.hdf5')

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