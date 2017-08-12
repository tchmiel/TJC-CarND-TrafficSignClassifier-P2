# Project 2 - Build a Traffic Sign Recognition Classifier ###


### Self-Driving Car Engineer Nanodegree ##
### Deep Learning ##

 ##### Author: Tom Chmielenski

 ##### Date: August 2017

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

[//]: # (Image References)

*TODO* : Insert Images

[image1]: ./output/num_examples_in_training.png "Number of Training Examples in Each Class"
[image2]: ./output/num_examples_in_validation.png "Number of Validation Examples in Each Class"
[image3]: ./output/num_examples_in_test.png "Number of Test Examples in Each Class"
[image4]: ./output/sample_of_class_images.png "Sample Images"
[image5]: ./output/image_preprocessing.png "Processing of Images"
[image6]: ./output/final_training_vs_accuracy_plots.png "Training vs. Validation"

[image11]: ./new_traffic_signs/do_not_enter.png "Do Not Enter"
[image12]: ./new_traffic_signs/end_of_speed_limit.png "End of Speed Limit"
[image13]: ./new_traffic_signs/keep_right.png "Keep Right"
[image14]: ./new_traffic_signs/right_turn.png "Right Turn"
[image15]: ./new_traffic_signs/roundabout.png "Roundabout"
[image16]: ./new_traffic_signs/speed_limit_20.png "Speed Limit 20"
[image17]: ./new_traffic_signs/speed_limit_60.png "Speed Limit 60"
[image18]: ./new_traffic_signs/stop_sign.png "Stop Sign"
[image19]: ./new_traffic_signs/straight_or_left.png "Straight Or Left"
[image110]: ./new_traffic_signs/traffic_sign.png "Traffic Sign"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


#### Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

*TODO* : Fix Github Link

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is **34799 images**
* The size of the validation set is **4410 images**
* The size of test set is **12630 images**
* The shape of a traffic sign image is **(32, 32,3)**
* The number of unique classes/labels in the data set is **43**.  This was computed by reading in the signnames.csv and determing the number of classIds within this file.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Here are three histograms which show how the distribution of images within each of the 43 sign classes.

![alt text][image1]

![alt text][image2]

![alt text][image3]

One observation to note here is that not all classes are represented equally.  Some classes have close to 2000 images, 
and some classes of signs have < 250.  

If I have the time, I will go back and create new images of the under represented classes, and try to augment
the existing images with new images, created from the original images by scaling, rotating or translating.  
Other augmentation techniques could be to warp the images, or variety its brightness.

Here is a first sample in each class to give us an idea of what the each class of sign looks like.
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It was suggested in our Project's Jupyter Notebook, to start with the LeNet-5 implemntation that we completed
in a previous step. The orignal LeNet-5 solution was set up for 28x28x1 MNINST images and only had 10 classes for output.
The Germain Traffice Sign Dataset contains 32x32x3 images and had 43 sign classifiers as output.

First, I replaced the loading of the MNIST code with code given to us to load the German Dataset.

Since the German dataset is already 32x32 pixels, we can remove the padding step used with the MNINST data.

Our version of the German dataset came with training, test, and validation data already sliced.  I did not
modify these sets as I want to make sure I achieve the 93% validation set accuracy  for a succesful project
submission.  It was given that we should expect a validation set accuracy of 89% after a successful implementation 
of a LeNet-5 implementation similar to the previous lab.  In fact, my initial implementation was slightly lower
with 87% validation set accuracy.

From Sermanet and LeCun's paper, [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), 
they improved their accuracy slightly from 98.9%8 to 99.17% after converting these images to grayscale. For my project, I figure I would
try to attempt this as well.

It was hinted at to normalize the data and a suggestion of a formula (pixel-128) / 128 should be used.
Preprocessing with grayscale and normalied images resulted in a 78.4% validation set accurary.  This was not
what I was expecting.  I read up on why this could be happening.

This "image=(image-128)/128" normalization will generate values between -1 and 1. 
It means a mean of 0 and a variance of 2. However, I found suggestions on the forums](https://discussions.udacity.com/t/normalization-and-accuracy-decrease/295403/20) that suggested
that the "image = np.array(image / 255.0 - 0.5 )" normalization will generate values between -0.5 and 0.5. 
It means a mean of 0 and a variance of 1.

The original mean of the dataset was 82.7, using the first normalization formula, it was reduced to
1.2, and after succesful implemenation of second normalization formula, the mean was closer to 0, around -0.2.

np.mean(X_train) = 82.677

np.mean(normalize) = 1.224

np.mean(better_normalize) = -0.178

With just this preprocessing change, I was now about 90.1% VA.


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]


I did not generate additional data, because 1) I was able to get my accuracy up to 97% with the existing data, and 
2) I did not have the time to augment the existing iamges.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final LeNet model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Flatten  			    | output 400												|
| Fully connected		| input 400, output 120        									|
| RELU					|												|
| Dropout					|	keep_prob = 0.5											|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| Dropout					|	keep_prob = 0.5											|
| Fully connected		| input 84, output 43        									|


You will have to experiment with the network architecture - add/reduce number of layers, filters etc - and experiment with different hyperparameters (learning rate, epochs, batch size, dropout rate etc), so that your model is able to fit the normalized training data.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer (this was the same optimizer found in the LeNet solution).  My final architecture looks like:

| Parameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Epochs         		| 40   							| 
| Batch Size            | 128 	|
| learning rate			|	0.00125										|
| mu					|	0										|
| sigma					|	0.1 |
| dropout			    |	keep_prod 50% |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Listed below is the series of trial run that I performed when trying to get my validation set accuracy.  I tried 
to valid parameters as were described in class, varying only one parameter trying to determine if adjusting
the value would improve my validation set accuracy.  See my commentary through-out these trials.

**Trial #1** - LeNet architecture similar to LeNet-Lab Solution, except using color images instead of grayscale
 Epoch 10, Batch_size 128, rate=0.001, mu=0, sigma = 0.1    VA= 87.0%

**Trial #2** - Use Grayscale and normalized images using image=(image-128)/128
 Epoch 10, Batch_size 128, rate=0.001, mu=0, sigma = 0.1    VA= 78.4%

Interesting, the validation accuracy went down - not what I was expecting
From, https://discussions.udacity.com/t/normalization-and-accuracy-decrease/295403/20, 
This "X_train=(X_train-128)/128" normalization will generate values between -1 and 1. It means a mean of 0 and a variance of 2.
The "X_train = np.array(X_train / 255.0 - 0.5 )" normalization will generate values between -0.5 and 0.5. It means a mean of 0 and a variance of 1.

np.mean(X_train) = 82.677
np.mean(normalize) = 1.224
np.mean(better_normalize) = -0.178

**Trial #3** - Use Grayscale and normalized images using image=np.array(X_train / 255.0 - 0.5 )
 Epoch 10, Batch_size 128, rate=0.001, mu=0, sigma = 0.1    VA= 90.1%
 
 Impressive - 

**Trial #4** - Increse Epoch to 100
 Epoch 100, Batch_size 128, rate=0.001, mu=0, sigma = 0.1    VA= 93.5%
 
Updating to 100 epochs. I get a 93.5% VA after about 93 epochs, let;s keep it at 100.

**Trial #5** - Double the Batch_size = 256
 Epoch 100, Batch_size 256, rate=0.001, mu=0, sigma = 0.1    VA= 93.2%
 
 Didn't accomplish much, try to lower down to 100

**Trial #6** - Decrease Batch_size = 100
 Epoch 100, Batch_size 100, rate=0.001, mu=0, sigma = 0.1    VA= 94.0%
 
 Slight increase, let's keep Batch=100
 
**Trial #7** - Decrease Learning Rate by 1/2
 Epoch 100, Batch_size 100, rate=0.0005, mu=0, sigma = 0.1    VA= 91.1%

**Trial #8** - Reverting back to rate=0.001, let's implement dropout 
  Added droput between Layer 3 and 4, with keep_prob = 0.8 on training, (leave keep_prob=1.0 on validataion)
  Epoch 100, Batch_size 100, rate=0.001, mu=0, sigma = 0.1, keep_drop = 0.8    VA= 94.0%
  
  After about 60 epochs, VA leveled off around 94.0%
  
**Trial #9** - Added 2nd droput between Layer 4 and 5
  Epoch 100, Batch_size 100, rate=0.001, mu=0, sigma = 0.1, keep_drop = 0.8    VA= 94.0%

**Trial #10** - Reduce EPOCHS to 65, increase batch_size to 128
  Epoch 65, Batch_size 128, rate=0.001, mu=0, sigma = 0.1, keep_drop = 0.8    VA= 95.1%

**Trial #11** - Decrease the keep_drop to 0.5
  Epoch 65, Batch_size 128, rate=0.001, mu=0, sigma = 0.1, keep_drop = 0.5    VA= 95.9%

**Trial #12** - Try reducing the rate by 1/4
  Epoch 65, Batch_size 128, rate=0.0075, mu=0, sigma = 0.1, keep_drop = 0.5    VA= 87.7%

**Trial #13** - Try increasing the rate by 1/4
  Epoch 65, Batch_size 128, rate=0.00125, mu=0, sigma = 0.1, keep_drop = 0.5   VA= 96.7%

**Trial #14** - Decreased the Batch size down to 100
  Epoch 65, Batch_size 100, rate=0.00125, mu=0, sigma = 0.1, keep_drop = 0.5   VA= 96.2%

**Trial #15** - Increase the batch size to 150
  Epoch 65, Batch_size 150, rate=0.00125, mu=0, sigma = 0.1, keep_drop = 0.5   VA= 94.4%

**Trial #16** - dropping batch size to 128, changing Epoch to 40
  Epoch 40, Batch_size 128, rate=0.00125, mu=0, sigma = 0.1, keep_drop = 0.5   VA= 96.2%

Finalizing on ths model architecture, as this 3% more than the 93% required.

**Final**  = Epoch 40, Batch_size 128, rate=0.00125, mu=0, sigma = 0.1, keep_drop = 0.5   VA= 96.1%
 

I have included two plots showing training vs validation losses and accuracies from the final trial run here:

![alt text][image6]


My final model results were:
* training set accuracy of      99.8%
* validation set accuracy of    96.1%
* test set accuracy of          94.3%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web and have converted to 32x32 pixels manually before testing:

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15] ![alt text][image16] 
![alt text][image17] ![alt text][image18] ![alt text][image19] 
![alt text][image110]

The stop sign image may be difficult to classify as it is further away and does not fill the entire image.
The last sign may be difficult to classify as it is not taken at a perendicular angle to the face of the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


References:

[The Black Magic of Deep Learning - Tips and Tricks for the practitioner](https://nmarkou.blogspot.co.at/2017/02/the-black-magic-of-deep-learning-tips.html)

[CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3/)
