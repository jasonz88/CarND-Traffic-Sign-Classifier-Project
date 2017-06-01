
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/trafficsign_0.png "Histogram"
[image2]: ./examples/trafficsign_1.png "Traffic Signs"
[image3]: ./examples/rgb2gray.png "Grayscale"
[image4]: ./examples/augmented.png "Augmented"
[image5]: ./examples/newimages.png "Traffic Signs New Image"
[image6]: ./examples/newimagepred.png "Traffic Sign New Image Prediction"
[image7]: ./examples/newimagewrong.png "Traffic Sign New Image Wrong"
[image8]: ./examples/featuremap30.png "Feature Map"
[image9]: ./examples/featuremap302.png "Feature Map"
[image10]: ./examples/featuremap100.png "Feature Map"
[image11]: ./examples/featuremap1002.png "Feature Map"


## Rubric Points
### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data.

![alt text][image1]
![alt text][image2]



### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because a typical photo has the colors and brightness limited to a certain range. So to achieve high accuracy we need to increases contrast for lower contrast image. Converting the image to grayscale and then apply a histogram normalization is the method to equalize the brightness levels of photos.

There are some more reasons [here](http://stackoverflow.com/questions/12752168/why-we-should-use-gray-scale-for-image-processing); for CNN, I also tried using color images with normalization. the result didn't beat grayscale. Computational wise grayscale is better too.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because we need to increases contrast for lower contrast image.

I decided to generate additional data because the distribution of training dataset for each class of traffic sign is not even. max / min count is > 10 (2010 vs 180); so for class with less training examples it is easy to be unrepresented thus cause overfitting to the training set. So adding more data to each class so that they have similar count of training example can solve this issue.

To add more data to the the data set, I used the keras.preprocessing.image.ImageDataGenerator because it allows various transformation based on current images, eg. rotation, width / heigth shift, shear stress, zoom, flip etc.

Here is an example of an original image and an augmented image:

![alt text][image4]

The difference between the original data set and the augmented data set:
From the image we can see that augmented data set has similar features represented as original but in different angle / size / location.


My final model is a variation of the LeNet architecture consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray normalized image   				| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x12	|
| RELU					|												|
| MaxPool 2x2        	| 1x1 stride, valid padding, outputs 27x27x12	|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 23x23x20	|
| RELU					|												|
| MaxPool 2x2        	| 1x1 stride, valid padding, outputs 22x22x20	|
| Fully connected		| Input 22x22x20=9680 output 480				|
| RELU					|												|
| Fully connected		| Input 480 output 84           				|
| RELU					|												|
| Fully connected		| Input 84 output 43       						|
| Softmax				|         				                        ||



To train the model, I used an AdamOptimizer with batch size 128, this is an experimental optimal batch size based on the hardware setup (1080Ti, 11G GPU memory); we want to utilize the GPU memory fully so that we can speed up by avoid overhead to move data in and out of GPU memory as well as achieve larger effetive step size in SGD; more details can be found [here](https://www.quora.com/Intuitively-how-does-mini-batch-size-affect-the-performance-of-stochastic-gradient-descent), 20 epochs and 0.001 learning rate since the model converges fast and Adam is able to control the learning rate.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.967
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose the [LeNet](https://github.com/udacity/CarND-LeNet-Lab) architecture; because it does a good job classifying the 10 class of hand written digits and the computation is relatively cheaper.
* What were some problems with the initial architecture?

It failed to get a validation accuracy greater than 0.93. The reasons include: 1. Dataset is not distributed evenly for each class thus the model can easily overfit to a few training examples; 2. the LeNet architecture has 2x2 stride for max pooling, this may lose some useful information; 3. LeNet is designed for a 10 class problem with hand written digits which has relatively less feature to capture; we need a larger depth for weight sharing and a larger input size to the fully connected layer to be able to tell apart different classes.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The convolution layer 1 and 2 depth is increased to 12 and 20 respectively. Also the max pooling stride changed from [1, 2, 2, 1] to [1, 1, 1, 1]. The 1st fully connected layer accordingly changed to 9680 -> 480 followed by 480 -> 84 conversion in the 2nd fully connected layer. I did this in order to preserve more information for each image batch and give the network better ability to discern different classes. To prevent overfitting, I introduced a L2 regularization when calculating the loss function so that large weigthts and biases are punished. These adjustment results in an overall validation accuracy increased to 0.967 and test accuracy to 0.952; but this LeNet model failed to get > 0.5 accuracy on the new image data. To solve this issue, augmented training data is introduced to help the model generalize. Each class has 3000 training examples. This gives another boost of the validation accuracy as well as test accuracy. New image accuracy improved to 0.944.

I also tried the Inception architecture with 1x1 convolution.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray normalized image   				| 
| SubLayer 1-1: Input -> Convolution 1x1    	| 1x1 stride, same padding, outputs 32x32x16	|
| SubLayer 1-2: Input -> Convolution 1x1 -> Convolution 3x3    	| 1x1 stride, same padding, outputs 32x32x16	|
| SubLayer 1-3: Input -> Convolution 1x1 -> Convolution 5x5    	| 1x1 stride, same padding, outputs 32x32x16	|
| SubLayer 1-4: Input -> MaxPool 3x3 -> Convolution 1x1    	| 1x1 stride, same padding, outputs 32x32x16	|
| Inception 1           | concatenate sublayers, add depths, output 32x32x64        |
| RELU					|												|
| SubLayer 2-1: Inception 1 -> Convolution 1x1    	| 1x1 stride, same padding, outputs 32x32x32	|
| SubLayer 2-2: Inception 1 -> Convolution 1x1 -> Convolution 3x3    	| 1x1 stride, same padding, outputs 32x32x32	|
| SubLayer 2-3: Inception 1 -> Convolution 1x1 -> Convolution 5x5    	| 1x1 stride, same padding, outputs 32x32x32	|
| SubLayer 2-4: Inception 1 -> MaxPool 3x3 -> Convolution 1x1    	| 1x1 stride, same padding, outputs 32x32x32	|
| Inception 2           | concatenate sublayers, add depths, output 32x32x128        |
| Fully connected					| Input 32x32x128=131072 output 700											|
| RELU					|												|
| Dropout               | Keep Prob 0.5                                 |
| Fully connected		| Input 700 output 43       									|
| Softmax				|         				                        ||



the architecture has lots of benefits: it uses all the convolution sizes; 1x1 convolution reduces dimension and is computationally cheap; it also create a deeper network without stacking more layers [\[1\]](https://www.reddit.com/r/MachineLearning/comments/3oln72/1x1_convolutions_why_use_them/cvyxood/); the dropout layer also prevents overfitting effectively. Inception with 1x1 convolution also gives 0.936 validation accuracy and 0.949 test accuracy. New image accuracy is 0.83.

I use LeNet for submission as it converges faster and computes faster with a smaller model size.
* Which parameters were tuned? How were they adjusted and why?
I tuned the learning rate, the beta for L2 regularization term coefficient, batch size as well as the augmented training data size. They are adjusted based on the validation accuracy, converge speed, effective step size, degree of overfitting etc.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer works well because it enables feature detection with space invariance and size invarianze because of different convolution size, layers of convolution and weight sharing. Dropout layer prevents overfitting by removing connections ramdomly so that huge weights causing overfitting can be avoided.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
Because the traffic sign dataset and MNIST dataset are both image classification problems with symbolic contents.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final model approaches 100% in training accuracy and has > 0.95 accuracy in both validation and testing data set. so it covers all the training data well and leave some ability to generalize to other unseen images for testing.

 

### Test a Model on New Images

Here are some German traffic signs that I found on the web:

![alt text][image5]

Those images might be difficult to classify because the portion containing the traffic signs are in general larger than those in the training and validation dataset. This specifically can cause some confusion when it comes to numbers in speed limits.

Here are the results of the prediction:


![alt text][image6]


The model was able to correctly guess 17 of the 18 traffic signs, which gives an accuracy of 0.944. This compares favorably to the accuracy on the test set. The test accuracy on new image is very close to the test accuracy on the test set. 0.944 vs 0.957. So the model trained is not overfitting to the training set.

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

For the only misclassified image:

|    Class                  |      Probability           |
|:---------------------:|:---------------:| 
| Speed limit (30km/h) | 100.00% |
| Speed limit (100km/h) | 0.00% |
| Speed limit (80km/h) | 0.00% |
| Speed limit (120km/h) | 0.00% |
| Stop | 0.00% ||

![alt text][image7]

We can see speed limit 30 vs 100 km/h was the top 2 prediction.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)


30km conv1![alt text][image8] 30km conv2![alt text][image9]
100km conv1![alt text][image10] 100km conv2![alt text][image11]

Here we are comparing the 30 km vs 100 km speed limits; From the feature map we can see for the 1st convolution layer output, edges in the number and circle contour are activated. But for the 2nd convolution layer output only a few pixels or cluster of pixels are activated. And the activation for the 2nd convolution layer are quite similar for 30 km vs 100 km speed limits. this explains why the 100 km speed limit is misclassfied as 30 km.




