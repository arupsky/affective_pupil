import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class FCNN_Re:
	"""docstring for CNN"""
	def __init__(self, x_train, y_train, config={}):
		self.config = config

		# n_timesteps, n_outputs = x_train.shape[1], y_train.shape[1]
		n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

		self.model = Sequential()
		firstTime = True
		for count in self.config["layers"]:
			if firstTime:
				# self.model.add(Dense(count, input_dim=n_timesteps, activation='relu'))
				self.model.add(Dense(count, input_shape=(n_timesteps, n_features), activation='relu'))
				firstTime = False
			else:
				self.model.add(Dense(count, activation='relu'))
		self.model.add(Flatten())
		self.model.add(Dense(n_outputs, activation='softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	def trainModel(self, trainFeatures, trainLabels, validateFeatures, validateLabels):
		x_train = trainFeatures
		y_train = trainLabels
		x_validate = validateFeatures
		y_validate = validateLabels
		
		self.model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.config["epochs"], batch_size=self.config["batch_size"], verbose=self.config["verbose"])
		return self.model

	def trainModel(self, trainFeatures, trainLabels):
		x_train = trainFeatures
		y_train = trainLabels
		
		self.model.fit(x_train, y_train, epochs=self.config["epochs"], batch_size=self.config["batch_size"], verbose=self.config["verbose"])
		return self.model
		