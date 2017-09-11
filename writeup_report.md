# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/speed15.gif "Speed 15"
[image2]: ./assets/speed20.gif "Speed 20"
[image3]: ./assets/speed25.gif "Speed 25"
[image4]: ./assets/left.png "Left"
[image5]: ./assets/center.png "Center"
[image6]: ./assets/right.png "Right"
[image7]: ./assets/biased-left.png "Biased Left"
[image8]: ./assets/biased-right.png "Biased Right"
[image9]: ./assets/original.png "original"
[image10]: ./assets/flipped.png "flipped"
[image11]: ./assets/opposite-direction.png "opposite direction"
[image12]: ./assets/cropped.png "cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the vehicle in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results
* `assets.ipynb` creating reference images
* `speed*.mp4` video when the vehicle drive autonomously each speed

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, and I set up a default value of `set_speed = 20` due to satisfy not too slow, and avoid leaving the road, the vehicle can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network using by Keras. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of convolution neural networks based on [NVIDIA's paper](https://arxiv.org/abs/1604.07316). First, The input data, whose shape is `(160, 320, 3)`, is normalized in the model using a Keras lambda layer, and it is cropped by 70px of upper side and 25px of lower side. After that, the model contains CNN with 5x5 filter sizes, stride 2, and depths 24, 36, and 48. Also, the model is similar with previous CNN, but it repeats twice with 3x3 filter sizes, stride 1, and depths 64. Next, the model consists fully connected with 100, 50, 10 and 1. Additionally, I apply RELU activation functions on the model with CNN layers. You can see below:

|input: 160x320x3|
|:---:|
|normalize: (x / 255.0) - 0.5|
|crop: ((70,25), (0,0))|
|conv layer: 5x5x24, stride 2|
|RELU|
|conv layer: 5x5x36, stride 2|
|RELU|
|conv layer: 5x5x48, stride 2|
|RELU|
|conv layer: 3x3x64, stride 1|
|RELU|
|conv layer: 3x3x64, stride 1|
|RELU|
|flatten|
|fully connected: 100|
|fully connected: 50|
|fully connected: 10|
|fully connected: 1|

```python
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 2. Model parameter tuning

The model used an ADAM optimizer on Keras library, so the learning rate was not tuned manually, and if you want to know what default values are, look at [here](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L401).

```python
model.compile(loss='mse', optimizer='adam')
```

Also, I apply a `tuned` parameter, which is forced steering angles depends on where the camera is located. In this model, the `tuned` parameter is forced to be `0.15`.

```python
train_generator = generator(train, batch_size=32, tuned=0.15)
valid_generator = generator(valid, batch_size=32, tuned=0.15)
```

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using by forced steering angles, which I mentioned the `tuned` parameter. I tried to apply 0.1, 0.15, and 0.2 that 0.15 showed best performance when the model demonstrated in autonomous driving.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use [NVIDIA's autonomous driving model](https://arxiv.org/abs/1604.07316), I thought this model might be appropriate because it proved by experiment, then I would be able to decide that I should make a training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

```python
from sklearn.model_selection import train_test_split
train, valid = train_test_split(lines, test_size=0.2)
```

I found that my first model had huge gap of mean squared error between training set and validation set, and often go to out of the road when I run the simulator. This is because my model did not know how a vehicle can recover toward a center of the road when it is located in biased position. I could solve this problem that I used left and right positioned images with forced steering parameters.

And most of corners in the simulator were left-turns. These problem was occurred by insufficient data, so I have to create training data for solving problems. I will describe in [next section](https://github.com/GzuPark/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md#2-creation-of-the-training-set--training-process).

The final step was to run the simulator to see how well the vehicle was driving around track one. Depends on the max speed of a vehicle show like below:

| Speed 15            | Speed 20            | Speed 25            |
|:-------------------:|:-------------------:|:-------------------:|
| ![speed 15][image1] | ![speed 20][image2] | ![speed 25][image3] |

I choose the speed 20, the vehicle is able to drive autonomously around the track without leaving the road and swaying.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is examples image of center lane driving: Left, Center, and Right camera views:

| Left            | Center            | Right            |
|:---------------:|:-----------------:|:----------------:|
| ![Left][image4] | ![Center][image5] | ![Right][image6] |

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to be able to go toward a center of the road when it is located at outside of the road. Here is the examples:

| Biased Left            | Biased Right            |
|:----------------------:|:-----------------------:|
| ![biased left][image7] | ![biased right][image8] |

To augment the data set, I also flipped images and angles thinking that this would avoid biased turned data on simulator. For example, here is images that have then been flipped and before:

| Original            | Flipped             |
|:-------------------:|:-------------------:|
| ![original][image9] | ![flipped][image10] |

Also, to make a variety of data, I drove one lap with an opposite direction looks like below:

| ![opposite direction][image11]  |
|---------------------------------|

After the collection process, I had 9497 number of data points. I then preprocessed this data by `generator()` function. [models.py 20-67](https://github.com/GzuPark/CarND-Behavioral-Cloning-P3/blob/84b894a92164359f140066598d810ba69cba98b0/model.py#L20)

I randomly shuffled the data set and put 20% of the data into a validation set.

During the training, I cropped the image by 70px of upper side and 25px of lower side:

| Original            | Cropped             |
|:-------------------:|:-------------------:|
| ![original][image9] | ![flipped][image12] |


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. For this work, I used batch normalization that training and validation set were [normalized](https://github.com/GzuPark/CarND-Behavioral-Cloning-P3/blob/a9dd8dddbf3d8e058ef192150af0fe5d8e3c774e/model.py#L70) by [every batch size](https://github.com/GzuPark/CarND-Behavioral-Cloning-P3/blob/a9dd8dddbf3d8e058ef192150af0fe5d8e3c774e/model.py#L79). The ideal number of epochs was 5 compared with 2, 8, and 10. I used an ADAM optimizer so that manually training the learning rate wasn't necessary. [here](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L401)

### Conclusions and Future Directions

In this project, I focus on autonomous driving safely, so I could apply verified convolutional neural network with well preprocessed data, which I created by various ways. However, the result had showed little swayed driving when it started.

I would like to augment the model, so I should think about below:

* I could not success when the speed is 30. For this, I need to modify model and preprocess.
* In this model, it decides a steering angle with only one image. But, a driving situation is sequential environment with past few images. Therefore, it would apply with recurrent neural network such as LSTM or GRU.
* I did not think about memory, but I can do convert from RGB to grayscale, crop images before flipping, and resize images such as 64x64 for performance.
