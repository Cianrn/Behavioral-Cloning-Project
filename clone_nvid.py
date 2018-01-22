import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle
import json

samples = []

with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

with open('./driving_log_track1_lap.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

del(samples[0])

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
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					filename = filename.split('\\')[-1]
					current_path = './IMG_comb/' + filename
					image = cv2.imread(current_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					images.append(image)
					
				correction = 0.15
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				angle_left = measurement + correction
				measurements.append(angle_left)
				angle_right = measurement - correction
				measurements.append(angle_right)

			augmented_images, augmented_measurements = [], []
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)		    
				augmented_images.append(cv2.flip(image, 1))
				augmented_measurements.append(measurement*-1.0)


			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield shuffle(X_train, y_train)				

batch_size = 32
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

ch, row, col = 3, 160, 320
model = Sequential()
model.add(Lambda(lambda x: (x/255) -0.5, input_shape = (row, col, ch)))
model.add(Cropping2D(cropping = ((55, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = 51768, validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 2, verbose =1)

model.save('model_nvid_one.h5')
outfile = open('./model_nvid_one.json', 'w') # of course you can specify your own file locations
json.dump(model.to_json(), outfile)
outfile.close()
model.save_weights('model_nvid_weights.h5')

print(history_object.history.keys())
fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
fig.savefig('test_val_acc1.png')
