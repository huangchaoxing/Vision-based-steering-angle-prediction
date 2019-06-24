# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:53:12 2019

@author: HP
"""

import os 
import cv2

image_list=list(os.walk('images/'))[0][2]

n=0
new_image_id=[]
for dir_ in image_list :
    
    frame_num=int(dir_[5:-4])
    if frame_num%3==0:
        new_image_id.append(frame_num)
        
new_image_id=sorted(new_image_id)        
for i in new_image_id:
    frame=cv2.imread('images/'+'frame'+str(i)+'.jpg')
    frame=frame[40:,:]
    cv2.imwrite('frames/'+'frame'+str(i//3)+'.jpg',frame)
        
        