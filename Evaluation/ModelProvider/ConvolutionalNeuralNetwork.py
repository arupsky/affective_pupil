from sklearn.model_selection import train_test_split
import numpy as np
from numpy import mean
from numpy import std
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
import time

class ConvolutionalNeuralNetwork:
	
	def __init__(self, trainFeatures, trainLabels, epochs = 100):
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels
		self.epochs = epochs
		self.log = ""
		# print(trainFeatures.shape)
		self.trainFeatures.reshape(self.trainFeatures.shape[0], self.trainFeatures.shape[1], 1)
		self.confusionMatrices = []
		self.yTrue = []
		self.yPred = []
		self.trainModel()

	def printEvaluation(self):
		print("Report : Convolutional Neural Network")
		print("feature shape : ", self.trainFeatures.shape, ", label shape : ", self.trainLabels.shape)
		print('Accuracy: %.3f%% (+/-%.3f)' % (self.efficiency, self.error))
		
	def trainModel(self):

		x_train_temp, x_test, y_train_temp, y_test = train_test_split(self.trainFeatures, self.trainLabels, test_size=0.33, random_state=42)
		x_train, x_validate, y_train, y_validate = train_test_split(x_train_temp, y_train_temp, test_size=0.33, random_state=42)

		

		n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
		# print("n_timesteps:", n_timesteps, "n_outputs:", n_outputs)

		batch_size = 32
		verbose = 0

		scores = []
		experimentCount = 1


		# print("train shape : ", x_train.shape, ",",y_train.shape)
		# print("validate shape : ", x_validate.shape)
		# print("test shape : ", x_test.shape)
		# print("CNN Training started...")

		self.logPrint(["train shape : ", x_train.shape, ",",y_train.shape])
		self.logPrint(["validate shape : ", x_validate.shape])
		self.logPrint(["test shape : ", x_test.shape])
		self.logPrint(["CNN Training started..."])
		self.logPrint(["Number of experiments ", experimentCount])
		startTime = time.time()
		for i in range(experimentCount):
			self.logPrint(["Running experiment # ", (i+1)])
			model = Sequential()
			model.add(Conv1D(filters=64, kernel_size=3, activation=tf.nn.relu6, input_shape=(n_timesteps,n_features)))
			model.add(Conv1D(filters=64, kernel_size=3, activation=tf.nn.relu6))
			model.add(Dropout(0.5))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Conv1D(filters=32, kernel_size=3, activation=tf.nn.relu6))
			model.add(Conv1D(filters=32, kernel_size=3, activation=tf.nn.relu6))
			model.add(Dropout(0.5))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Flatten())
			model.add(Dense(100, activation='relu'))
			model.add(Dense(n_outputs, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.epochs, batch_size=batch_size, verbose=verbose)
			_, accuracy = model.evaluate(x_test, y_test, verbose=0)
			prediction = model.predict(x_test)
			self.yTrue.append(y_test)
			self.yPred.append(prediction)
			scores.append(accuracy * 100)
			predictedLabels = tf.argmax(prediction,axis=1)
			realLabels = tf.argmax(y_test,axis=1)
			confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
			self.confusionMatrices.append(confusionMatrix)
			print("matrix ", confusionMatrix)

		self.efficiency,self.error = mean(scores),std(scores)
		self.logPrint(["CNN Training ended. Execution time : ", (time.time() - startTime)])
		self.logPrint(["Average efficiency ", self.efficiency, " (+/-)", self.error])

	def logPrint(self, log):
		parts = [str(lg) for lg in log]
		for part in parts:
			self.log = self.log + part
		self.log = self.log + "\n"
		print(log)

		