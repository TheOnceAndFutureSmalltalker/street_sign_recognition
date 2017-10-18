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
The model actually ran just fine with inital settings.  I ran the model dozens of times however to experiment with different aspects of training.  The results did not seem too sensitive to learning rate except when accuracy become very high, then a smaller learning rate was more beneficial.  Therefor, I included a decaying learning rate to attain highest accuracy possible.   

Batch size did seem to have quite an effect on accuracy.  128 did not produce results as well as 64.  I did not go lower than 64 thinking the sample size would be too low and accuracy would not converge.  I did not go higher than 256 for fear of running out of memory.  

I experimented some with image manipulation - rotation mostly.  This did not produce good results.  Very experimental.

### Final Model
The last saved model ended up with a validation accuracy 0.989.  This was achieved with the LeNet architecture described above, a batch size of 64, a learning rate starting at 0.0015 and decaying by 5% with each epoch.  I ran 20 epochs which was usually more than enough to get accuracy above 0.98.  A dropout rate of 0.5 was used to prevent overfitting to the training data.

## Testing
The model was tested against provided test data and additional images acquired form internet.

### Testing Dataset
The final model described above was run on the test data set just once for an accuracy of 0.941.

## Additional Images

I acquired two different sets of 5 images each from the internet.  One set are jpg and photos of real signs on the street. These have trees and clouds in background, are not necessarily centered, and might be skewed at an angle. The next set are png and are drawings of the signs like you might find in Adobe Illustrator for example.

These images are grabbed from Google images. I had to keep referencing signnames.csv and a Wikipedia page on German Road Signs to make sure I had the correct signs and correctly labeled! Both sets have two of the same signs, the traffic signal sign and the pedestrians sign.

The signs were of varying sizes and the png images had 4 channels. I had to account for this in loading the data. After laoding and resizing, the 10 images are displayed along with the y vectors showing the true labels against which the predictions will be compared.

The images are shown below in their original size (prior to processing) along with the actual and predicted feature value.

#### JPG Images
| Image | Actual | Pred |
|-------|--------|------|
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/speedlimit30.jpg" /> | 1 | 1 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/pedestrians.jpg" /> | 27 | 18 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/trafficsignals.jpg" /> | 26 | 1 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/novehicles.jpg" /> | 15 | 26 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/generalcaution.jpg" /> | 18 | 3 |

#### PNG Images
| Image | Actual | Pred |
|-------|--------|------|
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/speedlimit60.png" /> | 3 | 3 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/pedestrians.png" /> | 27 | 27 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/trafficsignals.png" /> | 26 | 26 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/bumpyroad.png" /> | 22 | 22 |
| <img src="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/download/gostraightorright.png" /> | 36 | 36 |

### Prediction Analysis

The png results were perfect which is not surprising since these are mostly ideal renditions of the signs! The jpg set results were 20% accurate, not good! This is understandable since the model was trained on centered, non-skewed, non-obstructed views of the signs.

### Softmax Analysis

A printout of the softmax analysis was performed showing top five probabilities for each of my ten images.  Its a little hard to read and I think a graph or just formatting the floats to 3 decimal places would be better.  I prefer the argmax on the logits since it gives the actual prediction.  The softmax results are provided on the jupyter notebook and report.html.

## Final Thoughts

It is disappointing that the jpg "real world" examples did not do better. Obviously, this model is not adequate for road use since the jpg images represent how road signs would appear to a camera mounted on a car.

What is needed is a better data set to train on. A dataset where all images are captured from a car's camera - inlcude various angles, various backgrounds, are partially obstructed, etc. Training a model on such a dataset would probably require much more effort than what was expended on this lab. Probably a more robust architecture as well.

I like the idea of using 32 X 32 images since most road signs will be viewed from a distance and therefor only constitute a small piece of a larger image.

I did not get to the section on displaying the network state images, but I plan to do this later as I think it is interesting and revealing about what is actually going on.


