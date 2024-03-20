import torch                  
import torch.nn as nn    
from PIL import Image          
import torch.nn.functional as F 
import torchvision.transforms as transforms    

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
    
model = ResNet9(3, 38)
print(type(model))
model = torch.load('plant-disease-model-complete.pth', map_location=torch.device('cpu'))
model.eval()

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
