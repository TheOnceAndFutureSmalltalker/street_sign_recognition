# Traffic Sign Recognition
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Data Set Summary & Exploration
The dataset is German Traffic Signs.  The dataset provided in a zip was already pickled and all images are at 32 X 32 X 3, eventhough the original images are larger.  The images have already been cropped and the sign is centered with adequate padding between the sign and the edge of image for edge detection methods.

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

#### Note
The video in the lab mentions that the data set does not have a validation set and to use a split from train.  However, when I unzipped the file, it did have a valiation set.  We caI set it up so that I can switch back and forth betwen the validation set provided and one split from training set to see if there are any changes in results.

### Data Summary

I used simple python to calculate basic summary of data as it was read in from the source files:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 X 32 X 3
* The number of unique classes/labels in the data set is 43

However, when validaton set was optionally defined as a split of training, then the training set was 27839 images and the validaton set was 6960 images.

### Data Visualization

I used numpy and matplotlib to plot the number of occurances of each sign in the training dataset.  You can see that some signs are ten times more prevalent than others.

<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/instances_count.png" />

I then found 5 images at random from training set and displayed them to get an idea of what the training images looked like.  The name of the signs are also provided. While the signs are centered, fill most of the image, and viewed from straight on, they appear dark and grainy.

Image | Name | Index
------|------|------
<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/training_yield.png" /> | Yield | 13
<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/training_general_caution.png" /> | General caution | 18
<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/training_turn_right_ahead.png" /> | Turn right ahead | 33
<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/training_end_of_no_passing.png" /> | End of no passing | 41
<img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/training_speed_limit_70.png" /> | Speed limit (70km/h) | 4

## Data Pre-processing

The data is preprocessed first by converting the type to int32 from ubyte. The data is then normalized using (pixel - 128) / 128 so that each value is in interval (-1, 1). The data conversion was necessary because the ubyte will not go negative.

The training set was then randomly shuffled. This is not entirely necessary here since it will be shuffled again for each epoch.

Finally a function is added for optionally manipulating the images with rotate and flip.

A sample of before and after data was printed out to verify the operation and can be seen on the notebook.

## Model Architecture

My final model consisted of a 5 layer LeNet described as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1     	| 5 X 5  convolution, ReLU activation followed by 2 X 2 pooling for an output of 14 X 14 X 6 	|
| Layer 2					|	5 X 5 convolution, ReLU activation, followed by 2 X 2 pooling for an output of 5 X 5 X 16 |
| Layer 3 | flattened input to single dimension of 400, ReLU activaton, final output of 120 |
| Layer 4 | fully connected layer, ReLU, final output of 84 |  				
| Dropout	| optional dropout applied here	|
| Layer 5	| fully connected layer final output of 43, which are the logits/prediction |        									|

## Model Training

### Batches
The model is trained by dividing the training set into batches.  Each image in a batch is run through the model, then loss is calculated for the entire batch, and then run through back propagation.  The optimizer is the Adam optimizer as this generally gets better results than SGD. The batch size is configurable.  This pipeline is described below:

### Pipeline
1. use LeNet to calculate the logits, 
2. calculate cross entropy with softmax and one hot encoding
3. calculate the loss
4. back propagate to update parameters using Adam optimizer

### Epochs
Each run through the entire training data set is an epoch and the number of epochs is configurable.  The model is run against the validation set for a measure of accuracy at the end of each epoch.  

### Dropout
Dropout can be included or excluded and the keep probability is configurable.

### Learning Rate
The learning rate is also configurable and may be static across all epochs or it may decay with each epoch by dividing the previous learning rate by a divisor which is also configurable.  I tried using some of the learning rate decay functions provided by TensorFlow but found them difficult.

### Stop Training
The training code includes early exit tests for two conditions 1) a very low accuracy rate that is not improving, 2) an accuracy rate that is very high > 0.99.

### Model Save
The model is saved at the end of a successful run.  Much experimentation was done without saving the model however.  I only started saving the model once I was close to a solution.

### Training Approach
The model actually ran just fine with inital settings.  I ran the model dozens of times however to experiment with different aspects of training.  The results did not seem too sensitive to learning rate except when accuracy become very high, then a smaller learning rate was more beneficial.  Therefor, I included an decaying learning rate to attain highest accuracy possible.   

Batch size did seem to have quite an effect on accuracy.  128 did not produce results as well as 64.  I did not go lower than 64 thinking teh sample size would be too low and accuracy would not converge.  I did not go higher than 256 for fear of running out of memory.  

I experimented some with image manipulation - rotation mostly.  This did not produce good results.  Very experimental.

### Final Model
The last saved model ended up with a validation accuracy 0.989.  This was achieved with a learning rate starting at 0.0015 and decaying by 5% with each epoch.  I ran 20 epochs which was usually more than enough to get accuracy above 0.98.  A dropout rate of 0.5 was used to prevent overfitting to the training data.

## Testing
The model was tested agains provided test data and additional images acquired form internet.

### Testing Dataset
The final model described above was run on the test data set just once for an accuracy of 0.941.

## Additional Images

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

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


