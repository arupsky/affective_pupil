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
import random

class FullyConnectedNetwork2:
	
	def __init__(self, trainFeatures, trainLabels, epochs = 100, random_state = 10):
		self.random_state = random_state

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
		print("Report : Fully Connected Neural Network")
		print("feature shape : ", self.trainFeatures.shape, ", label shape : ", self.trainLabels.shape)
		print('Accuracy: %.3f%% (+/-%.3f)' % (self.efficiency, self.error))

	def splitEven(self, features, labels, test_size=.33, random_state=46):
		
		random.seed(random_state)

		positiveFeatures = []
		neutralFeatures = []
		negativeFeatures = []

		positiveLabels = []
		neutralLabels = []
		negativeLabels = []


		total = labels.shape[0]
		targetTrainLength = int(total * (1-test_size))
		targetClassSize = int(targetTrainLength/3) + 1

		print("target class size", targetClassSize)

		x_train = []
		y_train = []
		x_test = []
		y_test = []


		for i in range(total):
			if np.array_equal(labels[i], [1,0,0]):
				negativeFeatures.append(features[i])
				negativeLabels.append(labels[i])
			elif np.array_equal(labels[i], [0,1,0]):
				neutralFeatures.append(features[i])
				neutralLabels.append(labels[i])
			else:
				positiveFeatures.append(features[i])
				positiveLabels.append(labels[i])

		# for negative
		random.seed(random_state)
		indexList = list(range(len(negativeLabels)))
		traiIndexList = random.sample(indexList, targetClassSize)
		for i in range(len(negativeLabels)):
			if i in traiIndexList:
				x_train.append(negativeFeatures[i])
				y_train.append(negativeLabels[i])
			else:
				x_test.append(negativeFeatures[i])
				y_test.append(negativeLabels[i])

		# for neutral
		random.seed(random_state)
		indexList = list(range(len(neutralLabels)))
		traiIndexList = random.sample(indexList, targetClassSize)
		for i in range(len(neutralLabels)):
			if i in traiIndexList:
				x_train.append(neutralFeatures[i])
				y_train.append(neutralLabels[i])
			else:
				x_test.append(neutralFeatures[i])
				y_test.append(neutralLabels[i])

		# for positive
		random.seed(random_state)
		indexList = list(range(len(positiveLabels)))
		traiIndexList = random.sample(indexList, targetClassSize)
		for i in range(len(positiveLabels)):
			if i in traiIndexList:
				x_train.append(positiveFeatures[i])
				y_train.append(positiveLabels[i])
			else:
				x_test.append(positiveFeatures[i])
				y_test.append(positiveLabels[i])

		print("lengths", len(positiveLabels), len(neutralLabels), len(negativeLabels))
		return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
		
	def trainModel(self):

		x_train_temp, x_test, y_train_temp, y_test = self.splitEven(self.trainFeatures, self.trainLabels, test_size=0.33, random_state=self.random_state)
		x_train, x_validate, y_train, y_validate = self.splitEven(x_train_temp, y_train_temp, test_size=0.33, random_state=self.random_state)
		

		n_timesteps, n_outputs = x_train.shape[1], y_train.shape[1]
		print("n_timesteps:", n_timesteps, "n_outputs:", n_outputs)

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
		self.logPrint(["FCNN Training started..."])
		self.logPrint(["Number of experiments ", experimentCount])
		# print("Real labels ", y_test)
		startTime = time.time()
		for i in range(experimentCount):
			self.logPrint(["Running experiment # ", (i+1)])
			model = Sequential()
			model.add(Dense(20, input_dim=9, activation='relu'))
			model.add(Dense(5, activation='relu'))
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
		self.logPrint(["FCNN Training ended. Execution time : ", (time.time() - startTime)])
		self.logPrint(["Average efficiency ", self.efficiency, " (+/-)", self.error])

	def logPrint(self, log):
		parts = [str(lg) for lg in log]
		for part in parts:
			self.log = self.log + part
		self.log = self.log + "\n"
		print(log)

		