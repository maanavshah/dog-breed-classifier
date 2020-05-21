# Dog Breed Classifier in PyTorch

This is a repo for the Dog Breed Classifier Project  in Udacity Nanodegree

It is implemented by using PyTorch library.

**Udacity's original repo is [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)**

You can find the blog for this project [here](https://medium.com/@maanavshah/dog-breed-classifier-using-cnn-f480612ac27a)


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI  Nanodegree! In this project, you will learn how to build a pipeline that  can be used within a web or mobile app to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification  and localization, you will make important design decisions about the  user experience for your app.  Our goal is that by completing this lab,  you understand the challenges involved in piecing together a series of  models designed to perform various tasks in a data processing pipeline.   Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.  Your imperfect solution will nonetheless create a fun  user experience!



## Import Datasets

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)



## CNN Structures (Building a model on my own)

(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

activation: relu

(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(dropout): Dropout(p=0.3)

(fc1): Linear(in_features=6272, out_features=512, bias=True)

(dropout): Dropout(p=0.3)

(fc2): Linear(in_features=512, out_features=133, bias=True)

**Explanation:** First 2 conv layers I've applied kernel_size of 3 with stride 2, this will lead to downsize of input image by 2. after 2 conv layers, maxpooling with stride 2 is placed and this will lead to downsize of input image by 2. The 3rd conv layers is consist of kernel_size of 3 with stride 1, and this will not reduce input image. after final maxpooling with stride 2, the total output image size is downsized by factor of 32 and the depth will be 128. I've applied dropout of 0.3 in order to prevent overfitting. Fully-connected layer is placed and then, 2nd fully-connected layer is intended to produce final output_size which predicts classes of breeds.

-----

​	Accuracy has been achieved up to **16%** with **20 epochs**





## Transfer Learnings

Used **Resnet50** for transfer learnings



Accuracy has been achieved up to **81%** with **30 epochs**


## Reference

![Test Accuracy](https://github.com/maanavshah/dog-breed-classifier/blob/master/my_images/test_accuracy.png)



## Result

The six dogs that were sampled to check the algorithm were correctly identified as dogs. The breeds of 5 of 6 were accurate too


![Test Dog Image](https://github.com/maanavshah/dog-breed-classifier/blob/master/my_images/dog1.png)

![Test Dog Image](https://github.com/maanavshah/dog-breed-classifier/blob/master/my_images/dog2.png)


The humans were also identified as human and a dog breed predicted — incidentally both were predicted as Dogue_de_bordeaux

![Test Human Image](https://github.com/maanavshah/dog-breed-classifier/blob/master/my_images/human1.png)

![Test Human Image](https://github.com/maanavshah/dog-breed-classifier/blob/master/my_images/human2.png)
