import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class CNN:
	"""docstring for CNN"""
	def __init__(self, trainFeatures, trainLabels, validateFeatures, validateLabels, modelOption):
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels
		self.validateFeatures = validateFeatures
		self.validateLabels = validateLabels
		# self.epochs = epochs
		# self.verbose = verbose
		# self.batch_size = batch_size
		self.modelOption = modelOption

	def trainModel(self):
		x_train = self.trainFeatures
		y_train = self.trainLabels
		x_validate = self.validateFeatures
		y_validate = self.validateLabels
		n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
		model = Sequential()
		# model.add(Conv1D(filters=10, kernel_size=5, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), input_shape=(n_timesteps,n_features)))
		model.add(Conv1D(filters=self.modelOption["layer1_num_filters"], kernel_size=self.modelOption["layer1_kernel_size"], activation=tf.nn.relu6, input_shape=(n_timesteps,n_features)))
		model.add(Dropout(self.modelOption["layer1_dropout"]))
		model.add(MaxPooling1D(pool_size=self.modelOption["layer1_pool_size"]))
		model.add(Conv1D(filters=self.modelOption["layer2_num_filters"], kernel_size=self.modelOption["layer2_kernel_size"], activation=tf.nn.relu6))
		model.add(Dropout(self.modelOption["layer2_dropout"]))
		model.add(MaxPooling1D(pool_size=self.modelOption["layer2_pool_size"]))
		model.add(Flatten())
		# model.add(Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(n_outputs, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.modelOption["epochs"], batch_size=self.modelOption["batch_size"], verbose=self.modelOption["verbose"])
		return model

	# def trainModel(self):
	# 	x_train = self.trainFeatures
	# 	y_train = self.trainLabels
	# 	x_validate = self.validateFeatures
	# 	y_validate = self.validateLabels
	# 	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	# 	model = Sequential()

	# 	for i in range(len(self.modelOption["conv_layers"])):
	# 		if i == 0:
	# 			model.add(Conv1D(filters=self.modelOption["conv_layers"][i]["num_filters"], kernel_size=self.modelOption["conv_layers"][i]["kernel_size"], activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), input_shape=(n_timesteps,n_features)))
	# 		else:
	# 			model.add(Conv1D(filters=self.modelOption["conv_layers"][i]["num_filters"], kernel_size=self.modelOption["conv_layers"][i]["kernel_size"], activation=tf.nn.relu6))
	# 		model.add(Dropout(self.modelOption["conv_layers"][i]["dropout"]))
	# 		model.add(MaxPooling1D(pool_size=self.modelOption["conv_layers"][i]["pool_size"]))

	# 	model.add(Flatten())

	# 	regularizer = True
	# 	for denseSize in self.modelOption["dense_layers"]:
	# 		if regularizer:
	# 			model.add(Dense(denseSize, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
	# 			regularizer = False
	# 			continue
	# 		model.add(Dense(denseSize, activation=tf.nn.relu6))


	# 	# model.add(Conv1D(filters=10, kernel_size=5, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), input_shape=(n_timesteps,n_features)))
	# 	# model.add(Conv1D(filters=self.modelOption["layer1_num_filters"], kernel_size=self.modelOption["layer1_kernel_size"], activation=tf.nn.relu6, input_shape=(n_timesteps,n_features)))
	# 	# model.add(Dropout(self.modelOption["layer1_dropout"]))
	# 	# model.add(MaxPooling1D(pool_size=self.modelOption["layer1_pool_size"]))
	# 	# model.add(Conv1D(filters=self.modelOption["layer2_num_filters"], kernel_size=self.modelOption["layer2_kernel_size"], activation=tf.nn.relu6))
	# 	# model.add(Dropout(self.modelOption["layer2_dropout"]))
	# 	# model.add(MaxPooling1D(pool_size=self.modelOption["layer2_pool_size"]))
	# 	# model.add(Flatten())
	# 	# model.add(Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
	# 	# model.add(Dense(10, activation='relu'))
	# 	model.add(Dense(n_outputs, activation='softmax'))
	# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# 	model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=self.modelOption["epochs"], batch_size=self.modelOption["batch_size"], verbose=self.modelOption["verbose"])
	# 	return model
		