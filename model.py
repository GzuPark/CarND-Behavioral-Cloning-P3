import csv
import cv2

# Read csv file which has information of image location and steering angles
lines = []
with open('../P3-data-4/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Split train dataset and valid dataset
train, valid = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32, tuned=0.2):
	"""
	This function provides shuffle X, y data every term by number of batch size.
	samples: 	input data (train or valid)
	batch_size: number of return data each term
	tuned:		forced add or substract steering angle for left or right cameras
	"""
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			end = offset + batch_size
			batch_samples = samples[offset:end]

			images = []
			angles = []
			for batch_sample in batch_samples:
				# Use center, left, and right camera images
				for i in range(3):
					source_path = batch_sample[i]
					name = source_path.split('/')[-1]
					current_path = '../P3-data-4/IMG/' + name

					image = cv2.imread(current_path)
					images.append(image)

					angle = float(batch_sample[3])
					# Apply forced steering angle toward center of road
					if i == 0:
						angles.append(angle)
					elif i == 1:
						angles.append(angle + tuned)
					elif i == 2:
						angles.append(angle - tuned)

			aug_images, aug_angles = [], []
			# Make flipped images
			# Track 1 has most of turned-left corners, so this work protects bias.
			for image, angle in zip(images, angles):
				aug_images.append(image)
				aug_angles.append(angle)
				aug_images.append(cv2.flip(image, 1))
				aug_angles.append(angle*-1.0)

			X = np.array(aug_images)
			y = np.array(aug_angles)

			yield shuffle(X, y)


train_generator = generator(train, batch_size=32, tuned=0.15)
valid_generator = generator(valid, batch_size=32, tuned=0.15)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# CNN: NVIDIA
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

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train),
					validation_data=valid_generator, nb_val_samples=len(valid),
					nb_epoch=5)

model.save('model.h5')
