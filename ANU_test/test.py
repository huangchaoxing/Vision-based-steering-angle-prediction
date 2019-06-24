#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:04:24 2019

@author: engn6528
"""


ITERATE_NUM=100
LR=1e-4
BATCH_SIZE=1
#from dataloader import Drivingset
from test_dataloader import load,Drivingset
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import time
from model_2s import NetworkDense,Net,Net_2st
#from dataloader import get_all_frames,log_to_list
import h5py
import random
from tensorboardX import SummaryWriter as writer
from math import pi
import pandas as pd
import cv2

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))




test_set=Drivingset()
test_loader=torch.utils.data.DataLoader(test_set,batch_size=1)


print(len(test_loader.dataset))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device='cpu'
print(device)

model=Net_2st()
results=[]

#    print(i)
model.load_state_dict(torch.load('flow.pth'))
model=model.cuda()
model.eval()
    
step=0
print("Ready !")

driving_result=[]
wheel=cv2.imread('steering_wheel.jpg')
smoothed_angle=0
with torch.no_grad():
        for i,data in enumerate(test_loader):
            
            image,dyn_img=data
            image,dyn_img = (image.to(device,dtype=torch.float),dyn_img.to(device,dtype=torch.float))
            
            outputs=model(image,dyn_img)
            theta=outputs.item()
            theta=theta/pi*180
            smoothed_angle += 0.2 * pow(abs((theta - smoothed_angle)), 2.0 / 3.0) * (theta - smoothed_angle) / abs(theta
                                       - smoothed_angle)
            frame=cv2.imread('frames/frame'+str(i)+'.jpg')
            rotate_wheel=rotate_bound(wheel,smoothed_angle)
            print(smoothed_angle)
            cv2.imshow('wheel',rotate_wheel)
            cv2.imshow('vis',frame)
            cv2.waitKey(1)
            driving_result.append([i,outputs.item()])
            
            
 
my_df = pd.DataFrame(driving_result)    
            
               
       
