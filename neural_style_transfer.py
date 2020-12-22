# -*- coding: utf-8 -*-

'''
Author - Mohd Aquib
'''
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

model = models.vgg19(pretrained=True).features

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features[:29]
    def forward(self,x):
        features = []
        for layer_num,layer in enumerate(self.model):
            x = layer(x)
            if(str(layer_num) in self.req_features):
                features.append(x)
        return features

def image_loader(path):
    image = Image.open(path)
    loader = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device,torch.float)

original_image = image_loader('/content/Nikhil.JPG')
style_image = image_loader('/content/style.jpg')

generated_image = original_image.clone().requires_grad_(True)

def calc_content_loss(gen_feat,orig_feat):
    content_l = torch.mean((gen_feat - orig_feat)**2)
    return content_l

def calc_style_loss(gen,style):
    batch_size,channel,height,width = gen.shape
    G = torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A = torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
    style_l = torch.mean((G-A)**2)
    return style_l

def calculate_loss(gen_features,orig_features,style_features):
    style_loss=content_loss=0
    for gen,con,style in zip(gen_features,orig_features,style_features):
        content_loss += calc_content_loss(gen,con)
        style_loss += calc_style_loss(gen,style)
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss

model = VGG().to(device).eval()

epoch = 7000
lr = 0.004
alpha = 8
beta = 70
optimizer = optim.Adam([generated_image],lr=lr)

for i in range(epoch):
    gen_features = model(generated_image)
    orig_features = model(original_image)
    style_features = model(style_image)

    total_loss = calculate_loss(gen_features,orig_features,style_features)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if(not(i%100)):
        print(total_loss)
        save_image(generated_image,'gen.png')

