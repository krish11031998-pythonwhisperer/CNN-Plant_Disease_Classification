import numpy as np
import pickle
import cv2
import os
from os import listdir
import keras
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Nadam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import rename_files as ref
import Tuning_CNN as tuning
import finalizing_CNN_design as fcd


class Plant_Images():

	def __init__(self, filename):
		self.filename = filename
		self.def_img_size = tuple((256, 256))
		self.width = 256
		self.height = 256
		self.depth = 3
		print("The label's have been binarized")
		self.image_list, self.label_list = self.image_list_generate()
		self.image_list_size = len(self.image_list)
		print("*" * 50)
		print("The size of the image list is : {}".format(self.image_list_size), end="\n" + "*" * 50 + "\n")
		print("The size of each image in the image list:")
		print(self.image_list[0], end="\n" + "/-" * 50 + "\n")
		print("The labels in the label list is:")
		print(self.label_list, end="\n" + "." * 50 + "\n")
		self.image_labels, self.n_classes = self.label_maker()
		print("The number of classes binarized {}".format(self.n_classes))
		self.data_preprocessing()

	def convert_img_to_array(self, img_dir):
		try:
			img = cv2.imread(img_dir)
			if img is not None:
				img_resized = img_to_array(cv2.resize(img, self.def_img_size))
				return img_resized
			else:
				return (np.array([]))

		except Exception as e:
			print("There was a error in converting the image: {}".format(e))

	def image_list_generate(self):
		print("[Start PROCESS] Starting to process the images")
		image_list, label_list = [], []
		try:
			for plant_disease in listdir(self.filename):
				if plant_disease == ".DS_Store":
					pass
				else:
					plant_disease_dir = self.filename + "/" + plant_disease
					print("[Processing] {}".format(plant_disease))
					for plant_img in listdir(plant_disease_dir)[:200]:
						print("[Processing] {}".format(plant_img))
						plant_img_dir = plant_disease_dir + "/" + plant_img
						if plant_img.endswith(".jpg") or plant_img.endswith(".JPG"):
							image_list.append(self.convert_img_to_array(plant_img_dir))
							label_list.append(plant_disease)
			return image_list, label_list
		except Exception as e:
			print("There was an error in the image_generator: {}".format(e))

	def label_maker(self):
		label_binarizer = LabelBinarizer()
		image_labels = label_binarizer.fit_transform(self.label_list)
		pickle.dump(label_binarizer, open('label_binarizer.pkl', 'wb'))
		n_classes = len(label_binarizer.classes_)
		return (image_labels, n_classes)

	def data_preprocessing(self):
		np_image_list = np.array(self.image_list, dtype=np.float16) / 225.0
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(np_image_list, self.image_labels,
																				test_size=0.3, random_state=42)
		self.image_aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
											height_shift_range=0.1, shear_range=0.2,
											zoom_range=0.2, horizontal_flip=True,
											fill_mode="nearest")


class Model(Plant_Images):

	def __init__(self, filename, name, layers, nons, params=None):
		super().__init__(filename=filename)
		self.num_of_layers = layers
		self.num_of_neurons = nons
		self.lr_init = 1e-3
		self.epochs = 5
		self.name_json = '.'.join([name, "json"])
		self.name_weights = '.'.join([name, "h5"])
		if params:
			self.params = params
			self.bs = params['batch_size']
			self.vers = 2
		else:
			self.bs = 32
			self.vers = 1

	def Input_Shape(self):

		if K.image_data_format() == 'channels_first':
			input_shape = (self.depth, self.height, self.width)
			chanDim = 1
		else:
			input_shape = (self.height, self.width, self.depth)
			chanDim = -1

		return input_shape, chanDim

	def CNN_model(self):

		self.model = Sequential()
		input_shape, chanDim = self.Input_Shape()

		if self.vers == 1:
			for i, num_neuron in zip(range(self.num_of_layers), self.num_of_neurons):
				if i == 0:
					self.model.add(Conv2D(num_neuron, (3, 3), padding='same', input_shape=input_shape))
					self.model.add(Activation('relu'))
					self.model.add(BatchNormalization(axis=chanDim))
					self.model.add(MaxPooling2D(pool_size=(2, 2)))
				else:
					self.model.add(Conv2D(num_neuron, (3, 3), padding='same'))
					self.model.add(Activation('relu'))
					self.model.add(BatchNormalization(axis=chanDim))
					self.model.add(MaxPooling2D(pool_size=(2, 2)))
					self.model.add(Dropout(0.25))

			self.model.add(Flatten())
			self.model.add(Dense(1024))
			self.model.add(BatchNormalization())
			self.model.add(Dropout(0.5))
			self.model.add(Dense(self.n_classes))
			self.model.add(Activation('softmax'))

			self.model.summary()
			self.model.compile(loss='binary_crossentropy',
							   optimizer=Adam(lr=self.lr_init, decay=self.lr_init / self.epochs), metrics=['accuracy'])


		elif self.vers == 2:
			activation = self.params['activation']
			dropout = self.params['dropout']
			optimizer = self.params['optimizer']
			for i, num_neuron in zip(range(self.num_of_layers), self.num_of_neurons):
				if i == 0:
					self.model.add(Conv2D(num_neuron, (3, 3), padding='same', input_shape=input_shape))
					self.model.add(Activation(activation))
					self.model.add(BatchNormalization(axis=chanDim))
					self.model.add(MaxPooling2D(pool_size=(2, 2)))
					self.model.add(Dropout(dropout))
				else:
					self.model.add(Conv2D(num_neuron, (3, 3), padding='same'))
					self.model.add(Activation(activation))
					self.model.add(BatchNormalization(axis=chanDim))
					self.model.add(MaxPooling2D(pool_size=(2, 2)))
					self.model.add(Dropout(dropout))

				self.model.add(Flatten())
				self.model.add(Dense(1024))
				self.model.add(BatchNormalization())
				self.model.add(Dropout(0.5))
				self.model.add(Dense(self.n_classes))
				self.model.add(Activation('softmax'))

			self.model.summary()
			optimizer = self.optimizer_design(optimizer)
			self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

			print("CNN designed and compiled!!, Ready to go !!")

	def optimizer_design(self, optimizer):
		if optimizer == "adam":
			opt = Adam(lr=self.lr_init, decay=self.lr_init / self.epochs)
		elif optimizer == "sgd":
			opt = SGD(lr=self.lr_init, decay=self.lr_init / self.epochs)
		elif optimizer == "nadam":
			opt = Nadam(lr=self.lr_init, decay=self.lr_init / self.epochs)

		return opt

	def fit_train(self):

		self.model_training = self.model.fit_generator(
			self.image_aug.flow(self.X_train, self.Y_train, batch_size=self.bs),
			validation_data=(self.X_test, self.Y_test),
			steps_per_epoch=(len(self.X_train) // self.bs),
			epochs=self.epochs, verbose=1)

	def save_classifier(self):
		save_model = self.model.to_json()
		with open(self.name_json, 'w') as json_file:
			json_file.write(save_model)
		self.model.save_weights(self.name_weights)
		print("Successfully saved the CNN model")

	def load_classifier(self):
		with open(self.name_json, 'r') as json_file:
			saved_model = json_file.read()
		self.model = keras.models.model_from_json(saved_model)
		self.model.load_weights(self.name_weights)
		self.model.compile(loss='binary_crossentropy',
						   optimizer=Adam(lr=self.lr_init, decay=self.lr_init / self.epochs), metrics=['accuracy'])
		print("[INFO] Sucessfully loaded the classifier")

	def running_classifier(self):
		if os.path.isfile(self.name_json) and os.path.isfile(self.name_weights):
			self.load_classifier()
		else:
			self.CNN_model()
			self.fit_train()
			self.evaluate_model_performance()
			self.save_classifier()

	def evaluate_model_performance(self):

		acc = self.model_training['acc']
		val_acc = self.model_training['val_acc']
		loss = self.model_training['loss']
		val_loss = self.model_training['val_loss']

		plt.figure(1)
		plt.plot(self.epochs, acc, 'b', label='Training accurarcy')
		plt.plot(self.epochs, val_acc, 'r', label='Validation accurarcy')
		plt.title('Training and Validation accurarcy')
		plt.legend()

		plt.figure(2)
		# Train and validation loss
		plt.plot(self.epochs, loss, 'b', label='Training loss')
		plt.plot(self.epochs, val_loss, 'r', label='Validation loss')
		plt.title('Training and Validation loss')
		plt.legend()
		plt.show()

		scores = self.model.evaluate(self.X_test, self.Y_test)
		print("The score for the model is : {}", format(scores))


if __name__ == "__main__":
	ref.file_check("/Users/krishnavenkatramani/Desktop/Plant_AI/PlantVillage-Dataset/raw/color")
	model_1 = Model(filename="/Users/krishnavenkatramani/Desktop/Plant_AI/PlantVillage-Dataset/raw/color",
					name="Plant_CNN_classifier", layers=5, nons=[32, 64, 64, 128, 128])
	model_1.running_classifier()
	tuning.tune()
	params = fcd.final_params()
	model_2 = Model(filename="/Users/krishnavenkatramani/Desktop/Plant_AI/PlantVillage-Dataset/raw/color",
					name="Plant_CNN_classifier_tuned", layers=5, nons=[32, 64, 64, 128, 128], params=params)
	model_2.running_classifier()
