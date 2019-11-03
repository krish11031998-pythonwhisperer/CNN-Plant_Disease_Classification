from MNIST_CNN_Model import *
# from GDA_upload import *
import keras
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Dense,Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from keras.datasets import fashion_mnist
import talos as ta

class MNIST_data():

	def __init__(self,dataset):
		(train_X,train_Y),(test_X,test_Y) = dataset.load_data()
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X
		self.test_Y = test_Y
		self.data_preprocessing()

	def data_preprocessing(self):
		self.train_X = (self.train_X.reshape(-1,28,28,1).astype('float32'))/255
		self.test_X = (self.test_X.reshape(-1,28,28,1).astype('float32'))/255

		self.train_Y = to_categorical(self.train_Y)
		self.test_Y = to_categorical(self.test_Y)



data = MNIST_data(fashion_mnist)

param = {
			'dense_neuron': [32,64,128,256],
			'dropout' : [0.2,0.3],
			'activation' : ['relu','sigmoid','elu'],
			'optimizer' : ['adam','Nadam','sgd'],
			'batch_size' : [64,128,256]
		}
x = np.concatenate((data.train_X,data.test_X),axis=0)
y = np.concatenate((data.train_Y,data.test_Y),axis=0)


def model_tune(x_train,y_train,x_val,y_val,params):

	dropout_num = params['dropout']
	dense_neuron = params['dense_neuron']

	model = Sequential()
	model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
	model.add(Activation(params['activation']))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(dropout_num))

	model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
	model.add(Activation(params['activation']))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(dropout_num))


	model.add(Flatten())
	model.add(Dense(dense_neuron))
	model.add(Activation(params['activation']))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer=params['optimizer'],metrics=['accuracy'])


	out = model.fit(x_train, y_train,
						 epochs=1,
						batch_size= params['batch_size'],
						 verbose=0,
						 validation_data=[x_val, y_val])

	return out,model

def tune():
	scan = ta.Scan(x,y,param,model_tune,experiment_name="Hyperparameter tuning")

