# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:26:24 2019

@author: HP
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:25:03 2019

@author: engn6528
"""
import numpy as np
#from dynamic_image import get_dynamic_image
import torchvision
import random
from torchvision import datasets
from torchvision import transforms 
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import pandas as pd
import random
from math import pi
import cv2


def sorted_data(data):
    id_list=[]
    for dir_ in data:
        frame_id=int(dir_[12:-4])
        id_list.append(frame_id)
    id_list=sorted(id_list)
    sorted_data=[]
    for i in id_list:
        sorted_data.append('frames/frame'+str(i)+'.jpg')
    return sorted_data    
    



class Drivingset(Dataset):
     def __init__(self,):
         
         self.data=glob.glob('frames/*')
         self.data=sorted_data(self.data)
         num=len(self.data)
         normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
         dyn_normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
              
      
         self.data=self.data[2:]
         self.trans=transforms.Compose([transforms.Resize([80,320]),
                                            transforms.ToTensor(),normalization])
         self.dyn_trans=transforms.Compose([transforms.Resize([80,320]),
                                          transforms.ToTensor(),dyn_normalization])
     def __getitem__(self,idx):
         image_dir=self.data[idx]
         image=Image.open(image_dir)
         image0=Image.open(self.data[idx-2])
         image1=Image.open(self.data[idx-1])
#         dynamic_image=Image.open(str(self.hist_num)+'_dyn/'+image_dir[:-4]+'dyn.jpg')
         flow_1=Image.open('flows_set/'+str(int(image_dir[12:-4]))+'.jpg')
         flow_2=Image.open('flows_set/'+str(int(image_dir[12:-4])-1)+'.jpg')

         image=self.trans(image) 
         image0=self.trans(image0)
         image1=self.trans(image1)
         frames=torch.cat((image,image1,image0),0) 
#         dynamic_image=self.dyn_trans(dynamic_image)
         flow_1=self.dyn_trans(flow_1)
         flow_2=self.dyn_trans(flow_2)
         flows=torch.cat((flow_1,flow_2),0)
         ####For single stream double input
#         data=torch.cat([image,dynamic_image],dim=0)
#         print(data.shape)
#         print(image.shape)
#         print(dynamic_image.shape)
         #########################################3
         return frames,flows
     def __len__(self):
         return len(self.data)



def load(batchsize):

    dataset=Drivingset()
    #validation_set=DogCatdataset(image_dir=img_dir,train=0)
    test_loader=torch.utils.data.DataLoader(dataset,batch_size=batchsize,num_workers=0)
   
    return test_loader


if __name__=='__main__':
    
    test_loader=load(30)    
    print(len(test_loader.dataset))
  
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # functions to show an image
    classes=np.linspace(0,4,1)
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        print(npimg.shape)
       
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
         
    # get some random training images
    dataiter = iter(test_loader)
    frames,dynamic_image=dataiter.next()
    
    # show images
    #imshow(torchvision.utils.make_grid(image))
 
    plt.figure(1)
    imshow(frames[-1][6:9])
    plt.figure(2)
    imshow(dynamic_image[-1][3:6])
    
    #print(classes[label])