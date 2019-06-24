# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:26:44 2019

@author: HP
"""

import cv2
import numpy as np
import pandas as pd
from math import pi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

#def simple_plot(prediction,groundtruth):
#    """
#    simple plot
#    """
#    # 生成画布
#    plt.figure(figsize=(8, 6), dpi=80)
#
#    # 打开交互模式
#    plt.ion()
#
#    # 循环
#    for index in range(100):
#        # 清除原有图像
#        plt.cla()
#
#        # 设定标题等
#        plt.title("steering angle trajectory")
#        plt.grid(True)
#
#        # 生成测试数据
#        x = np.linspace(-np.pi + 0.1*index, np.pi+0.1*index, 256, endpoint=True)
#        y_cos, y_sin = np.cos(x), np.sin(x)
#
#        # 设置X轴
#        plt.xlabel("frames")
#        plt.xlim(-4 + 0.1*index, 4 + 0.1*index)
##        plt.xticks(np.linspace(-4 + 0.1*index, 4+0.1*index, 9, endpoint=True))
#
#        # 设置Y轴
#        plt.ylabel("angle/degree")
#        plt.ylim(-1.0, 1.0)
##        plt.yticks(np.linspace(-1, 1, 9, endpoint=True))
#
#        # 画两条曲线
#        plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos")
#        plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin")
#
#        # 设置图例位置,loc可以为[upper, lower, left, right, center]
#        plt.legend(loc="upper left", shadow=True)
#
#        # 暂停
#        plt.pause(0.1)
#
#    # 关闭交互模式
#    plt.ioff()
#
#    # 图形显示
#    plt.show()
#    return


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
    return cv2.warpAffine(image, M, (300, 300), borderValue=(255,255,255))



plt.figure(figsize=(8, 6), dpi=80)
plt.ion()

ground_truth=pd.read_csv('DATA_0.txt',sep=' ', header=None)
ground_truth=ground_truth.values
gt_theta=[x[1] for x in ground_truth]
angles=pd.read_csv('3f2Fhill_trajectory.csv',header=None)
data=angles.values
angles=[x[1] for x in data]
#angles=[float(x[:-11]) for x in angles]
image_dir=[x[0] for x in data]
wheel=cv2.imread('steering_wheel.jpg')
i=0
font = cv2.FONT_HERSHEY_PLAIN
frame_list=[]
gth_list=[]
pre_list=[]
for theta in angles[35000-27500:]:
    g_theta=gt_theta[i+35000]
    gth_wheel=rotate_bound(wheel,g_theta)
    rotated_wheel=rotate_bound(wheel,theta/pi*180)
    if theta>0.08:
      print(theta/pi*180,'| ','right')
    elif theta<-0.08:  
      print(theta/pi*180,'| ','left') 
    else:
       print(theta/pi*180,'| ','straight') 
#    print(image_dir[i])
    camera=cv2.imread('data/'+image_dir[i+(35000-27500)])
    optical_flow=cv2.imread('flow_set/'+image_dir[i+(35000-27500)])
    frame_list.append(i)
    i=i+1
    cv2.namedWindow("camera")
    cv2.namedWindow("optical-flow")
#    cv2.namedWindow("pridict")
#    cv2.namedWindow('Groundtruth')
    cv2.resizeWindow('camera',(800,400))
    cv2.resizeWindow('optical-flow',(800,400))
    camera=cv2.resize(camera,(800,400))
    optical_flow=cv2.resize(optical_flow,(800,400))
    cv2.line(camera,(390,390),(int(390+100*np.sin(g_theta/180*pi)),int(390-100*np.cos(g_theta/180*pi))),(0,255,0),3)
    cv2.line(camera,(390,390),(int(390+70*np.sin(theta)),int(390-70*np.cos(theta))),(0,0,255),3)
    
    cv2.putText(camera, 'Ground truth:'+str(g_theta), (10,150), font, 1, (0, 255, 0), 1)
    cv2.putText(camera, 'prediction:'+str(theta/pi*180), (10,200), font, 1, (0, 0, 255), 1)
    cv2.putText(camera,'error:'+str(np.abs(theta/pi*180-g_theta)),(10,250),font,1,(0,255,255),1)
#     show_image=cv2.resize(image,(400,260))
    cv2.imshow('camera',camera)    
    cv2.imshow('optical-flow',optical_flow)
    cv2.imshow('groundth',gth_wheel)
    cv2.imshow('predict',rotated_wheel)
    cv2.waitKey(40)
    
    plt.cla()
    plt.xlabel("frames")
    plt.xlim(-1000 + (i-1), 1000 + (i-1))
    plt.ylim(-180,180)
#    print(g_theta)
    gth_list.append(g_theta)
    pre_list.append(theta/pi*180)
    plt.plot(frame_list, pre_list, "r-", linewidth=3.8, label="prediction")
    plt.plot(frame_list,gth_list , "g-", linewidth=3.0, label="ground-truth")
    

    plt.grid(True)   
    plt.legend(loc="upper left", shadow=True)

    plt.pause(0.05)

    