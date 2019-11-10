import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class CNN:
	"""docstring for CNN"""
	def __init__(self, trainFeatures, trainLabels, validateFeatures, validateLabels, epochs=500, verbose=0):
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels
		self.validateFeatures = validateFeatures
		self.validateLabels = validateLabels
		self.epochs = epochs
		self.verbose = verbose

	def trainModel(self):
		x_train = self.trainFeatures
		y_train = self.trainLabels
		x_validate = self.validateFeatures
		y_validate = self.validateLabels
		n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
		batch_size = 512
		model = Sequential()
		# model.add(Conv1D(filters=10, kernel_size=5, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), input_shape=(n_timesteps,n_features)))
		model.add(Conv1D(filters=10, kernel_size=5, activation=tf.nn.relu6, input_shape=(n_timesteps,n_features)))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Conv1D(filters=20, kernel_size=3, activation=tf.nn.relu6))
		model.add(MaxPooling1D(pool_size=4))
		model.add(Flatten())
		# model.add(Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(n_outputs, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.epochs, batch_size=batch_size, verbose=self.verbose)
		return model
		