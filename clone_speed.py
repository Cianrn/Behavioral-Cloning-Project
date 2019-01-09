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

path1 = 'Train/Log_T1_F1.csv' 
path2 = 'Train/Log_T1_F2.csv'  
path3 = 'Train/Log_T1_F3.csv' 
path4 = 'Train/Log_T1_F4.csv' 
path5 = 'Train/Log_T1_F5.csv' 
path6 = 'Train/Log_T1_F6.csv' 
path7 = 'Train/Log_T1_F7.csv' 
path8 = 'Train/Log_T1_F8.csv' 

# path9 = 'Train/Log_T2_F1.csv'  
# path10 = 'Train/Log_T2_F2.csv' 
path11 = 'Train/Log_T2_F3.csv' 
path12 = 'Train/Log_T2_F4.csv' 
path13 = 'Train/Log_T2_F5.csv' 
path14 = 'Train/Log_T2_F6.csv' 
path15 = 'Train/Log_T2_F7.csv'
path16 = 'Train/Log_T2_F8.csv'

path17 = 'Train/Log_T3_F1.csv'  
path18 = 'Train/Log_T3_F2.csv' 
path19 = 'Train/Log_T3_F3.csv' 
path20 = 'Train/Log_T3_F4.csv' 
path21 = 'Train/Log_T3_F5.csv' 
path22 = 'Train/Log_T3_F6.csv' 
path23 = 'Train/Log_T3_F7.csv'
path24 = 'Train/Log_T3_F8.csv'

path25 = 'Train/Log_T1_R1.csv'
path26 = 'Train/Log_T2_R1.csv'
path27 = 'Train/Log_T3_R1.csv'

paths = [path1, path2, path3, path4, path5, path6, path7, path8,
		path11, path12, path13, path14, path15, path16, path17, path18, path19, path20,
		path21, path22, path23, path24, path25, path26, path27]

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
					current_path2 = 'Train/IMG_T1_F2/' + filename 
					current_path3 = 'Train/IMG_T1_F3/' + filename 
					current_path4 = 'Train/IMG_T1_F4/' + filename
					current_path5 = 'Train/IMG_T1_F5/' + filename
					current_path6 = 'Train/IMG_T1_F6/' + filename 
					current_path7 = 'Train/IMG_T1_F7/' + filename
					current_path8 = 'Train/IMG_T1_F8/' + filename

					current_path9 = 'Train/IMG_T2_F1/' + filename
					current_path10 = 'Train/IMG_T2_F2/' + filename
					current_path11 = 'Train/IMG_T2_F3/' + filename
					current_path12 = 'Train/IMG_T2_F4/' + filename
					current_path13 = 'Train/IMG_T2_F5/' + filename 
					current_path14 = 'Train/IMG_T2_F6/' + filename
					current_path15 = 'Train/IMG_T2_F7/' + filename
					current_path16 = 'Train/IMG_T2_F8/' + filename

					current_path17 = 'Train/IMG_T3_F1/' + filename
					current_path18 = 'Train/IMG_T3_F2/' + filename
					current_path19 = 'Train/IMG_T3_F3/' + filename
					current_path20 = 'Train/IMG_T3_F4/' + filename
					current_path21 = 'Train/IMG_T3_F5/' + filename 
					current_path22 = 'Train/IMG_T3_F6/' + filename
					current_path23 = 'Train/IMG_T3_F7/' + filename
					current_path24 = 'Train/IMG_T3_F8/' + filename

					current_path25 = 'Train/IMG_T1_R1/' + filename
					current_path26 = 'Train/IMG_T2_R1/' + filename
					current_path27 = 'Train/IMG_T3_R1/' + filename

					if os.path.exists(current_path1) == True:
						image = cv2.imread(current_path1)
					elif os.path.exists(current_path2) == True:
						image = cv2.imread(current_path2)
					elif os.path.exists(current_path3) == True:
						image = cv2.imread(current_path3)
					elif os.path.exists(current_path4) == True:
						image = cv2.imread(current_path4)
					elif os.path.exists(current_path5) == True:
						image = cv2.imread(current_path5)
					elif os.path.exists(current_path6) == True:
						image = cv2.imread(current_path6)
					elif os.path.exists(current_path7) == True:
						image = cv2.imread(current_path7)
					elif os.path.exists(current_path8) == True:
						image = cv2.imread(current_path8)
					elif os.path.exists(current_path9) == True:
						image = cv2.imread(current_path9)	
					elif os.path.exists(current_path10) == True:
						image = cv2.imread(current_path10)	
					elif os.path.exists(current_path11) == True:
						image = cv2.imread(current_path11)
					elif os.path.exists(current_path12) == True:
						image = cv2.imread(current_path12)
					elif os.path.exists(current_path13) == True:
						image = cv2.imread(current_path13)
					elif os.path.exists(current_path14) == True:
						image = cv2.imread(current_path14)
					elif os.path.exists(current_path15) == True:
						image = cv2.imread(current_path15)	
					elif os.path.exists(current_path16) == True:
						image = cv2.imread(current_path16)
					elif os.path.exists(current_path17) == True:
						image = cv2.imread(current_path17)
					elif os.path.exists(current_path18) == True:
						image = cv2.imread(current_path18)
					elif os.path.exists(current_path19) == True:
						image = cv2.imread(current_path19)
					elif os.path.exists(current_path20) == True:
						image = cv2.imread(current_path20)
					elif os.path.exists(current_path21) == True:
						image = cv2.imread(current_path21)
					elif os.path.exists(current_path22) == True:
						image = cv2.imread(current_path22)
					elif os.path.exists(current_path23) == True:
						image = cv2.imread(current_path23)
					elif os.path.exists(current_path24) == True:
						image = cv2.imread(current_path24)
					elif os.path.exists(current_path25) == True:
						image = cv2.imread(current_path25)
					elif os.path.exists(current_path26) == True:
						image = cv2.imread(current_path26)
					else:
						image = cv2.imread(current_path27)	

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
