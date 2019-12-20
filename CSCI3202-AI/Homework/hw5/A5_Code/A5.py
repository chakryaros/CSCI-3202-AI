from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model



import numpy as np
import glob
from PIL import Image

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os



def getVGGFeatures(directory, layerName):
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)

	extractedFeatures = np.array([])
	# features = [ ]
	
	for f in os.listdir(directory):
		image_path = os.path.join(directory, f) 
		img = image.load_img(image_path)

		img = img.resize((224, 224))
		
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		# print("x ", x.shape)
		x = preprocess_input(x)
		ext_image = model.predict(x)
		
		ext_image = ext_image.flatten()
		
		ext_image = ext_image.reshape((-1, ext_image.shape[0]))
		ext_image = ext_image.astype('float32')
		# print("ext_image reshape", ext_image.shape)
		if extractedFeatures.shape[0] == 0:
			extractedFeatures = ext_image
		else :
			extractedFeatures = np.concatenate((extractedFeatures, ext_image),0)
	
	return extractedFeatures

def cropImage(image, x1, y1, x2, y2):
	# crop_img = image[y1:y2, x1:x2]
	img = Image.open(image).convert("L")

	cropped = img.crop((x1, y1, x2, y2))
	
	newImage = image
	newImage = newImage[10:]
	cropped.save("cropped/" + newImage)
	# cropped.show()
	return cropped

def standardizeImage(image, x, y):
	# img = Image.open(image)
	image_resize = image.resize((x, y))
	# image_resize.show()
	
	return image_resize
	

def preProcessImages(images, dim):

	crop_img = cropImage(images, dim[0], dim[1], dim[2], dim[3])
	image_resize = standardizeImage(crop_img, 60, 60)
	newImage = images
	newImage = newImage[10:]
	image_resize.save("resize/" + newImage)
	

	

def visualizeWeight(model):
	
	pass


def trainFaceClassifier(preProcessedImages, labels):
	

	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.30)

	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# normalizing the data to help with the training
	
	X_train/= 255
	X_test /= 255

	# one-hot encoding using keras' numpy-related utilities
	y_train = y_train.astype(int)
	# n_classes = np.max(y_train) + 1
	n_classes = 6
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)

	# my_init = K.random_normal(X_train.shape, dtype=int)

	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(32, kernel_initializer='random_normal', input_shape=(3600,)))
	model.add(Activation('relu'))                            
	model.add(Dense(6))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	model.summary()

	# training the model and saving metrics in history
	history = model.fit(X_train, y_train,
	          batch_size=128, epochs=100,
	          verbose=2,
	          validation_data=(X_test, y_test))

	predictions = model.predict(X_test)
	print('First prediction:', predictions[0])

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	

	return history


def trainFaceClassifier_VGG(extractedFeatures, labels):

	# the data, split between train and test sets
	X_train, X_test, y_train, y_test = train_test_split(extractedFeatures, labels, test_size=0.30)

	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	X_train/= 255
	X_test /= 255
	# one-hot encoding using keras' numpy-related utilities
	y_train = y_train.astype(int)
	# n_classes = np.max(y_train) + 1
	n_classes = 6
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)

	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(32, kernel_initializer='random_normal', input_shape=(100352,)))
	model.add(Activation('relu'))                            
	model.add(Dense(6))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	# training the model and saving metrics in history
	print(y_train.shape)
	print(y_test.shape)
	history = model.fit(X_train, y_train,
	          batch_size=128, epochs=100,
	          verbose=2,
	          validation_data=(X_test, y_test))
	model.summary()

	
	return history


def plot_loss(epochs, train_loss, validation_loss, title):

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

	epochs_range = range(epochs)

	plt.plot(epochs_range, train_loss, label='Training Loss')
	plt.plot(epochs_range, validation_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title(title)
	ax.set_ylabel("loss", fontsize=18)
	ax.set_xlabel("epochs", fontsize=18)
	plt.show()

def ExperimentFaceClassifier(directory, labels):

	preProcessedImages = np.array([])


	for f in os.listdir(directory):
		image_path = os.path.join(directory, f) 
		img = image.load_img(image_path)
		img = img.resize((224, 224))
		
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		
		ext_image = x
		ext_image = ext_image.astype('float32')
		
		if preProcessedImages.shape[0] == 0:
			preProcessedImages = ext_image
		else :
			preProcessedImages = np.concatenate((preProcessedImages, ext_image),0)

	
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.30)

	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)
	
	# one-hot encoding using keras' numpy-related utilities
	y_train = y_train.astype(int)
	
	n_classes = 6
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)


	print("before model",X_train.shape)
	model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='random_normal', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Dropout(0.25),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.25),
    Flatten(),
    Dense(6, activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

	model.summary()

	history = model.fit(X_train, y_train,
	          batch_size=128, epochs=10,
	          verbose=2,
	          validation_data=(X_test, y_test))

	predictions = model.predict(X_test)
	# print('First prediction:', predictions[0])

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return history

def ExperimentFaceClassifierVGG16(directory, labels, layerName):

	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	preProcessedImages = np.array([])

	for f in os.listdir(directory):
		image_path = os.path.join(directory, f) 
		img = image.load_img(image_path)
		img = img.resize((224, 224))
		
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		ext_image = model.predict(x)
		
		ext_image = ext_image.astype('float32')
		if preProcessedImages.shape[0] == 0:
			preProcessedImages = ext_image
		else :
			preProcessedImages = np.concatenate((preProcessedImages, ext_image),0)

	# break
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.30)
	
	# one-hot encoding using keras' numpy-related utilities
	y_train = y_train.astype(int)
	
	n_classes = 6
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)

	print("before model",X_train.shape)
	model = Sequential([
	Dense(6, kernel_initializer='random_normal'),
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Dropout(0.25),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(62, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.25),
    Flatten(),
    
    Dense(6, activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	

	history = model.fit(X_train, y_train,
	          batch_size=128, epochs=10,
	          verbose=2,
	          validation_data=(X_test, y_test))

	model.summary()

	predictions = model.predict(X_test)
	print('First prediction:', predictions[0])

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	

	return history

def experimentOnVGG16(directory):


	preProcessedImages = np.array([])


	for f in os.listdir(directory):
		image_path = os.path.join(directory, f) 
		img = image.load_img(image_path)
		img = img.resize((224, 224))
		
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		
		ext_image = x
		ext_image = ext_image.astype('float32')
		
		if preProcessedImages.shape[0] == 0:
			preProcessedImages = ext_image
		else :
			preProcessedImages = np.concatenate((preProcessedImages, ext_image),0)

	
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.30)

	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)
	
	# one-hot encoding using keras' numpy-related utilities
	y_train = y_train.astype(int)
	
	n_classes = 6
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)


	base_model = VGG16()
	model  = Sequential()
	for layer in base_model.layers[:-1]:
		model.add(layer)


	 # model.layers.pop()

	model.summary()

	# Freeze the layers 
	for layer in model.layers:
		layer.trainable = False

	# Add 'softmax' instead of earlier 'prediction' layer.
	model.add(Dense(6, activation='softmax'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	
	model.summary()

	history = model.fit(X_train, y_train,
	          batch_size=128, epochs=10,
	          verbose=2,
	          validation_data=(X_test, y_test))


	return history



if __name__ == '__main__':
	print("Your Program Here")


	for filename in glob.glob("uncropped/*.jpg"): #assuming gif
		
		f = filename
		img = f.split(',')
		
		img[4] = img[4].replace('.jpg', '')
		dim = [int(img[1]), int(img[2]), int(img[3]), int(img[4])]

		preProcessImages(filename, dim)
	   

	# part1

	# labels = [1, 2, 3, 4, 5, 6]
	preProcessedData = "resize/"
	preProcessed = np.array([])
	listData = []

	# for f in os.listdir(preProcessedData):
	for filename in glob.glob("resize/*.jpg"):
		listData.append(filename)
		img = Image.open(filename).convert("L")

		pixels = np.asarray(img)

		pixels = pixels.flatten()
		
		pixels = pixels.reshape((-1, pixels.shape[0]))
		pixels = pixels.astype('float32')
		# print(pixels)
		# print("should normalizing", pixels.shape)
		if preProcessed.shape[0] == 0:
			preProcessed = pixels
		else :

			preProcessed = np.concatenate((preProcessed,pixels), 0)
		
	# print(preProcessed)

	labels = np.zeros(len(listData))
	# print(labels)
	for idx, img in enumerate(listData):
		
		if img[:13] == "resize/bracco":
			labels[idx] = 0
		elif img[:13] == "resize/butler":
			labels[idx] = 1
		elif img[:13] == "resize/gilpin":
			labels[idx] = 2
		elif img[:13] == "resize/harmon":
			labels[idx] = 3
		elif img[:16] == "resize/radcliffe":
			labels[idx] = 4
		elif img[:13] == "resize/vartan":
			labels[idx] = 5
	
	
	# #part2
	
	history = trainFaceClassifier(preProcessed, labels)
	train_loss = history.history["loss"]
	validation_loss = history.history["val_loss"]

	plot_loss(100, train_loss, validation_loss, 'Training and Validation Loss')


	#part3

	extractedFeatures = getVGGFeatures('resize/', 'block4_pool')
	historyVGG = trainFaceClassifier_VGG(extractedFeatures, labels)
	train_loss_vgg = historyVGG.history["loss"]
	validation_loss_vgg = historyVGG.history["val_loss"]
	plot_loss(100, train_loss_vgg, validation_loss_vgg, 'Training and Validation Loss of VGG16')



	#part4

	##experiment 1 and 2
	create new function that call ExExperimentFaceClassifier to improve performance from part2

	history_exp = ExperimentFaceClassifier("resize/", labels)
	train_loss_exp = history_exp.history["loss"]
	validation_loss_exp = history_exp.history["val_loss"]
	plot_loss(10, train_loss_exp, validation_loss_exp, 'Training and Validation Loss Experiment')

	#experiment 3
	history_exp_VGG16 = experimentOnVGG16("resize/")
	train_loss_exp_Vgg16 = history_exp_VGG16.history["loss"]
	validation_loss_exp_Vgg16 = history_exp_VGG16.history["val_loss"]
	plot_loss(10, train_loss_exp_Vgg16, validation_loss_exp_Vgg16, 'Training and Validation Loss Experiment VGG16')
	


	##experiment 4
	#create new function that call ExExperimentFaceClassifier to improve performance from part2
	history_exp_VGG16 = ExperimentFaceClassifierVGG16("resize/", labels, 'block2_pool')

	train_loss_exp_Vgg16 = history_exp_VGG16.history["loss"]
	validation_loss_exp_Vgg16 = history_exp_VGG16.history["val_loss"]
	plot_loss(10, train_loss_exp_Vgg16, validation_loss_exp_Vgg16, 'Training and Validation Loss Experiment VGG16')
	


	
	




	






