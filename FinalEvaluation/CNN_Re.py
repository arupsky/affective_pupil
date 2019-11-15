import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class CNN_Re:
	"""docstring for CNN"""
	def __init__(self, x_train, y_train, modelOption):
		self.modelOption = modelOption
		n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
		self.model = Sequential()
		for i in range(len(self.modelOption["conv_layers"])):
			if i == 0:
				self.model.add(Conv1D(filters=self.modelOption["conv_layers"][i]["num_filters"], kernel_size=self.modelOption["conv_layers"][i]["kernel_size"], activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), input_shape=(n_timesteps,n_features)))
			else:
				self.model.add(Conv1D(filters=self.modelOption["conv_layers"][i]["num_filters"], kernel_size=self.modelOption["conv_layers"][i]["kernel_size"], activation=tf.nn.relu6))
			# model.add(Dropout(self.modelOption["conv_layers"][i]["dropout"]))
			self.model.add(MaxPooling1D(pool_size=self.modelOption["conv_layers"][i]["pool_size"]))

		self.model.add(Flatten())

		regularizer = True
		for denseSize in self.modelOption["dense_layers"]:
			if regularizer:
				self.model.add(Dense(denseSize, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
				regularizer = False
				continue
			self.model.add(Dense(denseSize, activation=tf.nn.relu6))

		self.model.add(Dense(n_outputs, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	def trainModel(self, trainFeatures, trainLabels, validateFeatures, validateLabels):
		x_train = trainFeatures
		y_train = trainLabels
		x_validate = validateFeatures
		y_validate = validateLabels
		self.model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.modelOption["epochs"], batch_size=self.modelOption["batch_size"], verbose=self.modelOption["verbose"])
		return self.model
		