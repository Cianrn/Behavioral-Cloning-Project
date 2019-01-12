import matplotlib
matplotlib.use('Agg')
import csv
import json
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

samples = []

def data_reader(paths):
	""" Read in data from excel sheets """
	for path in paths:
		with open(path) as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				# Delete start as the car is only taking off and not actual speed.
				# del(line[0:50])
				samples.append(line)

def remove_bias(samples, pct, bias):
	""" Remove zero bias """
	samples_new = samples
	samples_array = np.asarray(samples_new)
	steering_angles = samples_array[:,6]
	index = []
	for idx, ang in enumerate(steering_angles):
		random = np.random.uniform()
		if ((float(ang) >= bias[0] and float(ang) <= bias[1]) and random < pct): #(((float(ang) > -0.01) and (float(ang) < 0.01)) and random < pct):
			index.append(idx)

	for i in sorted(index, reverse = True):
		del(samples_new[i])

	return samples_new 

def brightness(image):
	""" Randomly change V-channel in HSV to vary brightness """
	img_br = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	random = np.random.randint(4)
	if random == 0:
		random_bright = 0.30 + np.random.uniform()
		img_br[:,:,2] = img_br[:,:,2]*random_bright
		
	img_br = cv2.cvtColor(img_br, cv2.COLOR_HSV2RGB)

	return img_br	

paths = 'Train/Log_T1_F1.csv' 

data_reader(paths)

# speeds = [float(speed[6]) for idx, speed in enumerate(samples)]
# fig = plt.figure()
# plt.hist(speeds, 100)
# plt.ylabel('Frequency')
# plt.xlabel('Speed')
# fig.savefig('hist1_Speed')

# # samples = remove_bias(samples, pct=.25, bias=[7,15])
# speeds = [float(speed[6]) for idx, speed in enumerate(samples)]
# fig = plt.figure()
# plt.hist(speeds, 100)
# plt.ylabel('Frequency')
# plt.xlabel('Speed')
# fig.savefig('hist2_Speed')

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
print(len(samples))
print(len(train_samples))

def generator(samples, batch_size):
	num_samples = len(samples)
	 
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			end = offset + batch_size
			batch_samples = samples[offset: end]

			images, measurements = [], []
			for batch_sample in batch_samples:
				for i in range(3):
					# center, left & right
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					filename = filename.split('\\')[-1]

					current_path1 = 'Train/IMG_T1_F1/' + filename 
					
					if os.path.exists(current_path1) == True:
						image = cv2.imread(current_path1)	

					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					# image = brightness(image)
					images.append(image)

				speed = float(batch_sample[6])
				measurements.append(speed)
				correction = -3
				measurements.append(speed + correction)
				measurements.append(speed + correction)

				augmented_images, augmented_measurements = [], []
				for image, veloc in zip(images,measurements):
					augmented_images.append(image)
					augmented_images.append(cv2.flip(image, 1))
					augmented_measurements.append(veloc)
					augmented_measurements.append(veloc)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield shuffle(X_train, y_train)

batch_size = 32
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

ch, row, col = 3, 160, 320
model = Sequential()
model.add(Lambda(lambda x: (x/255) -0.5, input_shape = (row, col, ch)))
model.add(Cropping2D(cropping = ((70, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu', name = 'conv1'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu', name = 'conv2'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu', name = 'conv3'))
model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'conv4'))
model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'conv5'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

samples_per_epoch = len(train_samples)*3*2
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch, validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 5)

model.save('model_nvid_speed.h5')
outfile = open('./model_nvid_speed.json', 'w') # of co
json.dump(model.to_json(), outfile)
outfile.close()
model.save_weights('model_nvid_speed_weights.h5')

# print(history_object.history.keys())
# fig = plt.figure()
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.ylabel('Mean Squared Error Loss')
# plt.xlabel('Epoch')
# fig.savefig('test_val_acc_speed1.png')
