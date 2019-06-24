import torch.nn as nn
import torch
import torchvision

'''
The light net need to be modified
'''
#checkpoint = torch.load('./model.h5', map_location=lambda storage, loc: storage)
#network_light= checkpoint['model']

class NetworkLight_24(nn.Module):

    def __init__(self):
        super(NetworkLight_24, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(24, 36, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 72, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=72*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
#        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
#        print(output.shape)
        output = output.view(output.size(0), -1)
#        output = self.linear_layers(output)
        return output 


class NetworkLight_9(nn.Module):

    def __init__(self):
        super(NetworkLight_9, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(9, 12, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(12, 24, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=24*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
#        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
#        print(output.shape)
        output = output.view(output.size(0), -1)
#        output = self.linear_layers(output)
        return output 

    
    
class NetworkLight_3(nn.Module):

    def __init__(self):
        super(NetworkLight_3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(12, 24, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=24*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
#        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
#        print(output.shape)
        output = output.view(output.size(0), -1)
#        output = self.linear_layers(output)
        return output     
    
    
    
    
class NetworkLight_6(nn.Module):

    def __init__(self):
        super(NetworkLight_6, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 12, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(12, 24, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=24*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
#        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
#        print(output.shape)
        output = output.view(output.size(0), -1)
#        output = self.linear_layers(output)
        return output       
    
    
    
    
    
    
    
class NetworkDense(nn.Module):

    def __init__(self):
        super(NetworkDense, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.ReLU(),
#            nn.BatchNorm2d(24),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
#            nn.BatchNorm2d(36),
            nn.Conv2d(64, 128, 5, stride=2),
             nn.ReLU(),
#            nn.BatchNorm2d(48),
            nn.Conv2d(128, 512, 3),
             nn.ReLU(),
#            nn.BatchNorm2d(64),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
           
#            nn.BatchNorm2d(64)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512* 18, out_features=1164),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1164, out_features=512),
            nn.ReLU(),
            
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
           
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
           
            nn.Linear(in_features=128, out_features=1)
            
           
        )
        
    def forward(self, input):  
#        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
#        print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
#        output=2*torch.atan(output)
        return output
    
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv0=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
#                                 nn.ReLU(), 
#                                 nn.BatchNorm2d(16))
##                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(32))                       
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(64),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(256),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
#                                 nn.Dropout2d(0.5),
                                 nn.BatchNorm2d(256),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        
        self.last=nn.Sequential(nn.Linear(23*23*256,1024),
                                nn.ReLU(),
                                
                                nn.Linear(1024,256),
                                nn.ReLU(),
                               
                                nn.Linear(256,1))

    def forward(self, x):
        #x=self.conv0(x)
#        x=self.conv0(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
#        print(x.shape)
        x=x.view(x.size(0),-1)
        
        output= self.last(x)
        return output   
    
    
    
class Extractor_gray(nn.Module):
    def __init__(self):
        super(Extractor_gray, self).__init__()
#        self.conv0=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
#                                 nn.ReLU(), 
#                                 nn.BatchNorm2d(16))
##                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(16))                       
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=16,out_channels=24,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(24),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=24,out_channels=36,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(36),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=36,out_channels=72,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(72),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=72,out_channels=128,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
#                                 nn.Dropout2d(0.5),
                                 nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        

    def forward(self, x):
        #x=self.conv0(x)
#        x=self.conv0(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
#        print(x.shape)
       
        return x    
    
class Extractor_rgb(nn.Module):
    def __init__(self):
        super(Extractor_rgb, self).__init__()
#        self.conv0=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
#                                 nn.ReLU(), 
#                                 nn.BatchNorm2d(16))
##                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=24,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(24))                       
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=24,out_channels=32,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(32),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(64),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
#                                 nn.Dropout2d(0.5),
                                 nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
#                                 nn.Dropout2d(0.5),
                                 nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        

    def forward(self, x):
        #x=self.conv0(x)
#        x=self.conv0(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
#        print(x.shape)
       
        return x        
    
########
# 8 FRAMES AND 3 FRAMES ARE DIFFERENT IN RGB_NET, RAMEMBER TO CHANGE THE SUB-MODEL and fc layers
########
    
class Net_2st(nn.Module):
    def __init__(self):
        super(Net_2st, self).__init__()
        self.dyn_net=NetworkLight_6()
       
        self.rgb_net=NetworkLight_9()
        self.controller=nn.Sequential(nn.Linear(24*4*19*2,256),
                                nn.ReLU(),
                                nn.Dropout(0.6),
                                nn.Linear(256,10),
                                nn.ReLU(),
                               
                                nn.Linear(10,1))
    def forward(self,x,y):
        x=self.rgb_net(x)
        y=self.dyn_net(y)
#        print(x.shape)
#        print(y.shape)
#        print(x.view(x.size(0),-1).shape)
#        print(y.view(y.size(0),-1).shape)
        z=torch.cat([x,y],1)
#        print(z.shape)
        z=self.controller(z)
        
        return z
