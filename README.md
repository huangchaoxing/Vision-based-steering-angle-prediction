# Vision-based-steering-angle-prediction
Semester project of ANU ENGN6528(computer vision)
## Background  
In this project, we use a end to end two-stream CNN model to predict the steering angle of a self-driving car. We first train the model on the dataset which is collected by [SullyChen](https://github.com/SullyChen) in LA, USA and then test it on the dataset which is collected on the campus of the Australian National University, Canberra, Australia.   
## Dependency 
Pytorch 1.0  
torchvision  
OpenCV(cv2)  

## Dataset
The dataset can be downloaded [here](https://github.com/SullyChen/driving-datasets)  
 
 ## Method  
 The model takes the RGB information and temporal information as the inputs of the two stream respectively and finally output the steering angle, which is a regression probelm. The temporal information can be optical-flow or [dynamic images](https://www.egavves.com/data/cvpr2016bilen.pdf).   
 The network architecture is shown below.Detalis can be referred to the technical report.  
