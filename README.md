# DeepWeeds_Classifier_WebApp
Recently, we published a paper "XYZ", can be found here: 

This is an neural network webapp visualizing the training of the network and testing accuracy ~ 99% accuracy.
The neural network uses pretrained EfficientNet_Lite4 and then trained to classify images of weeds.
It is built using Pytorch framework using Python as primary language.
The webapp is built using Flask.

## Dataset used :     
9 Category Weeds Dataset and the baseline of classification performance can be found here:     
https://github.com/AlexOlsen/DeepWeeds
And the original paper of the dataset was introduced by Alex Olsen et al., (https://www.nature.com/articles/s41598-018-38343-3)
## Neural Network used : 
EfficientNet family was introduced by a paper from Google Brain at https://arxiv.org/pdf/1905.11946.pdf
And codes for the EfficientNet family were hacked by Ross Wrightman. Thank Ross for his fantastic work to create valuable models for image classification tasks on PyTorch. We can find the codes of Ross here: https://github.com/rwightman/pytorch-image-models and https://github.com/rwightman/gen-efficientnet-pytorch
* You can download the trained weight of EfficientNet_Lite4 model [here](https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/release/EfficientNet_Lite4_SGD.pth) and other trained weights.   

## Flow:
* To reproduce our experiments in our paper, you can use NoteBook folder [here] (https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/Notebooks/)
* Predicted results of our models can be examed [here] (https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/Results/)
* Input image is fed and transformed using : [commons.py](https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/commons.py)     
* Inference is done by : [inference.py](htttps://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/inference.py) 
* Run on local web: [app.py] (https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp/blob/master/app.py) 

## Run on Ubuntu and MacOS, but not test on Windows - 
Make sure you have installed Python , Pytorch, Flask and other related packages, refer requirement.txt.

* _First download all the folders and files_     
`git clone https://github.com/linhduongtuan/DeepWeeds_Classifier_WebApp.git`     
* _Then open the command prompt (or powershell) and change the directory to the path where all the files are located._       
`cd DeepWeeds_Classifier_WebApp`      
* _Now run the following commands_ -        

`python app.py`     


This will firstly download the models and then start the local web server.

now go to the local server something like this - http://127.0.0.1:5000/ and see the result and explore.

### @creator - Duong Tuan Linh
