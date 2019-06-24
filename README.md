# Vision-based-steering-angle-prediction
[PROJECT REPORT](https://github.com/huangchaoxing/Vision-based-steering-angle-prediction/blob/master/report.pdf)
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
 The network architecture is shown below.Detalis can be referred to the [technical report](https://github.com/huangchaoxing/Vision-based-steering-angle-prediction/blob/master/report.pdf).  
![CNN model](https://github.com/huangchaoxing/Vision-based-steering-angle-prediction/blob/master/model.png)

## Result
#### Driving on campus:  
![gif](https://github.com/huangchaoxing/Vision-based-steering-angle-prediction/blob/master/demo.gif)

#### Full DEMO video:  
https://www.youtube.com/watch?v=7juEYI-gGKw&feature=youtu.be  

#### Root mean square error of different model(in degree)
For the  split of the datset, please refer to the report.   

| Model | City | Hill |
| ------------- | ------------- | ------------- |
| single stream single RGB | 19.04 | 23.89 |
| single stream 3 x RGB | 7.23 | 13.91 |
| 3 x RGB +1 x optical flow | 7.10 | 10.92 |
| 3x RGB +2 x optical flow | 6.09 | 9.73 |
| 3 x RGB +dynamic image of 3 frames| 6.93 | 10.15 |
| 3 x RGB +dynamic image of 5 frames| 5.99 | 11.51 |
#### Activation heatmap  
![hmp](https://github.com/huangchaoxing/Vision-based-steering-angle-prediction/blob/master/hmp.png)
## Run the code  
**NOTE: The code shall be slightly modified accroding to your own directory **   
#### Pre-processing
