#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:55:36 2019

@author: engn6528
"""
import cv2
from PIL import Image
import glob
import numpy as np
image_list = []


for i in range(0,1196):
    image0=cv2.imread('frames/frame'+str(i)+'.jpg')
    hsv=np.zeros_like(image0)
    hsv[...,1]=255
    image0=cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
    image1=cv2.imread('frames/frame'+str(i+1)+'.jpg')
    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    flow1=cv2.calcOpticalFlowFarneback(image0,image1,None,0.5,3,15,3,5,1.2,0)
    mag,ang=cv2.cartToPolar(flow1[...,0],flow1[...,1])
    hsv[...,0]=ang*180/np.pi/2
    hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb_flow=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('flows_set/'+str(i+1)+'.jpg',rgb_flow)
    print(i/63824)
