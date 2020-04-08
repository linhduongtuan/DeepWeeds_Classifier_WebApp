import io
import io
import PIL
import json
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch import optim
from copy import deepcopy
import torch.nn.functional as F
from geffnet import create_model
from torchvision import transforms

def get_tensor(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_model_fruits():
    model = create_model('tf_efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 120)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                      lr=0.001,momentum=0.9,
                      nesterov=True,
                      weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    CHECK_POINT_PATH = '/home/linh/Downloads/fruits/EfficientNet_B0_SGD.pth'
    checkpoint = torch.load(CHECK_POINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint['best_val_loss']
    best_acc = checkpoint['best_val_accuracy']
    model.eval()
    return model


with open('/home/linh/Downloads/fruits/cat_to_name.json') as f:
    cat_to_name_fruits = json.load(f)
with open('/home/linh/Downloads/fruits/class_to_idx.json') as f:
    class_to_idx_fruits = json.load(f)
with open('/home/linh/Downloads/fruits/labels.txt', 'r') as f:
    classes_fruits = f.readline()
    
model_fruits = get_model_fruits()
 
model_fruits.eval()

def get_fruit_name(image_bytes):
    fruit_names = []
    tensor = get_tensor(image_bytes)
    outputs = model_fruits.forward(tensor)
    
    # Probs
    probs = F.softmax(outputs, dim=1)
      
    # Top probs
    top_probs, top_labs = probs.topk(k=5)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    top_probs = top_probs  
    # Convert indices to classes
    idx_to_class_fruits = {val: key for key, val in class_to_idx_fruits.items()}
    top_labels = [idx_to_class_fruits[lab] for lab in top_labs]
    top_fruits = [cat_to_name_fruits[idx_to_class_fruits[lab]] for lab in top_labs]
    return top_probs, top_labels, top_fruits



def get_model_weeds():
    model = create_model('tf_efficientnet_lite4', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 9)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                      lr=0.001,momentum=0.9,
                      nesterov=True,
                      weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    CHECK_POINT_PATH = '/home/linh/Downloads/DeepWeeds/weights/EfficientNet_Lite4_SGD.pth'
    checkpoint = torch.load(CHECK_POINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint['best_val_loss']
    best_acc = checkpoint['best_val_accuracy']
    model.eval()
    return model



with open('/home/linh/Downloads/DeepWeeds/DeepWeeds_Classifier_WebApp/cat_to_name.json') as f:
    cat_to_name_weeds = json.load(f)
with open('/home/linh/Downloads/DeepWeeds/DeepWeeds_Classifier_WebApp/class_to_idx.json') as f:
    class_to_idx_weeds = json.load(f)
with open('/home/linh/Downloads/DeepWeeds/DeepWeeds_Classifier_WebApp/labels.txt', 'r') as f:
    classes_weeds = f.readline()
    
model_weeds = get_model_weeds()
 
model_weeds.eval()

def get_weed_name(image_bytes):
    weed_names = []
    tensor = get_tensor(image_bytes)
    outputs = model_weeds.forward(tensor)
    
    # Probs
    probs = F.softmax(outputs, dim=1)
      
    # Top probs
    top_probs, top_labs = probs.topk(k=5)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    top_probs = top_probs  
    # Convert indices to classes
    idx_to_class_weeds = {val: key for key, val in class_to_idx_weeds.items()}
    top_labels = [idx_to_class_weeds[lab] for lab in top_labs]
    top_weeds = [cat_to_name_weeds[idx_to_class_weeds[lab]] for lab in top_labs]
    return top_probs, top_labels, top_weeds




