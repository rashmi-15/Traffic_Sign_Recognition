import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models


from PIL import Image
import os

def load_model(path):
    model = models.resnet34(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 43)
    for param in model.parameters():
        param.requires_grad=True
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    return model

def load_transforms():
	return transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225]),
                         ])

def predict(net, transform, image):
	classes = ['Ahead only','Beware of ice/snow','Turn right ahead','Turn left ahead',
    'Children crossing', 'No vehicles', 'Dangerous curve right',
    'Double curve', 'End speed + passing limits', 'Speed limit (20km/h)', 'No passing veh over 3.5 tons', 
    'Right-of-way at intersection','General caution', 'Give_way','Go straight or left','Go straight or right', 'Veh > 3.5 tons prohibited',
    'Dangerous curve left', 'No passing', 'Bicycles crossing', 'End of speed limit (80km/h)', 'No entry', 'Pedestrians',
    'Priority road', 'Road narrows on the right', 'Road work', 'Keep right', 'Bumpy road', 'Slippery road', 
    'Speed limit (100km/h)', 'Speed limit (120km/h)', 'End no passing veh > 3.5','Speed limit (30km/h)','Speed limit (50km/h)',
    'Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)','Stop','Traffic signals',
    'Keep left', 'Roundabout mandatory', 'End of no passing','Wild animals crossing']

	input_tensor = transform(image).unsqueeze(0) 
	output = F.softmax(net(input_tensor), dim=1) 

	prob, pred = torch.max(output, 1)
	out_label = classes[pred]
	return out_label, prob.item() * 100