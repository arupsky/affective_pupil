from os import listdir
from os import makedirs
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import pickle

# model output node : 4 
# model input : length of data
# hidden layers : 2 

class ModelProvider(object):
	"""docstring for ModelProvider"""
	def __init__(self, featureFolder):
		super(ModelProvider, self).__init__()
		self.inputFolder = "FeatureDataProvider/Outputs/" + featureFolder+"/"
		self.outputFolderRoot = "ModelProvider/Outputs/"
		self.classCount = 3
		self.data = []
		self.models = []
		self.modelInfo = []

	def saveModels(self, fileName, model):
		keras_file = self.outputFolderRoot + self.outputFolderName +"/" + fileName.split(".")[0] + ".h5"
		tf.keras.models.save_model(model, keras_file)

	def createNewFolder(self):
		folders = [int(f.split('.')[0]) for f in listdir(self.outputFolderRoot) if not isfile(join(self.inputFolder, f))]
		if len(folders) == 0:
			self.outputFolderName = "0"
		else:
			self.outputFolderName = str(max(folders) + 1)

		makedirs(self.outputFolderRoot + self.outputFolderName)

	def loadAndTrain(self, epochs):
		self.createNewFolder()
		dataFiles = [f.split('.')[0]+".json" for f in listdir(self.inputFolder) if (isfile(join(self.inputFolder, f)) and ".json" in f)]
		
		for fileName in dataFiles:
			with open(join(self.inputFolder, fileName)) as file:
				loadedData = json.load(file)
				self.trainModelMatchingData(loadedData, fileName, epochs)

			# break
		with open(self.outputFolderRoot + self.outputFolderName + "/info.json", "w") as file:
			json.dump(self.modelInfo, file)

	def trainModelMatchingData(self, loadedData, fileName, epochs):
		random.seed(100)
		indexList = random.sample(range(len(loadedData["features"])),int(len(loadedData["features"])/4))
		inputNodeCount = len(loadedData["features"][0])
		nodeConfig = self.getNodeConfig(inputNodeCount)
		features = self.getFeaturesTrain(loadedData["features"], inputNodeCount, indexList)
		labels = self.getLabelsTrain(loadedData["labels"], indexList)
		featuresTest = self.getFeaturesTest(loadedData["features"], inputNodeCount, indexList)
		labelsTest = self.getLabelsTest(loadedData["labels"], indexList)
		model = self.getModelWithConfig(nodeConfig)
		model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
		model.fit(features, labels, epochs=epochs)
		evalData = model.evaluate(featuresTest, labelsTest)
		self.models.append(model)
		
		self.saveModels(fileName, model)
		modelInfo = {}
		modelInfo["epochs"] = epochs
		modelInfo["testFeatures"] = featuresTest.tolist()
		modelInfo["testLabels"] = labelsTest.tolist()
		modelInfo["fileName"] = fileName.split(".")[0]
		modelInfo["efficiency"] = float(evalData[1])
		modelInfo["loss"] = float(evalData[0])
		self.modelInfo.append(modelInfo)
		# self.modelInfo.append({
		# 	"testFeatures":featuresTest.tolist(),
		# 	"testLabels":labelsTest.tolist(),
		# 	# "model":model,
		# 	"fileName":fileName,
		# 	"evaluationData":model.evaluate(featuresTest, labelsTest)
		# })
		# print(nodeConfig, len(labels), ",", len(features))
		print(evalData)

	def getModelWithConfig(self, nodeConfig):
		model = keras.Sequential()
		# # Adds a densely-connected layer with 9 units to the model:
		model.add(keras.layers.Dense(nodeConfig[1], input_dim = nodeConfig[0], activation='relu'))
		# # Add another:
		model.add(keras.layers.Dense(nodeConfig[2], activation='relu'))
		# # Add a softmax layer with 5 output units:
		model.add(keras.layers.Dense(nodeConfig[3], activation='softmax'))

		return model

	def getNodeConfig(self, inputNodeCount):
		nodeCounts = [inputNodeCount]
		if inputNodeCount < 5:
			nodeCounts.append(10)
			nodeCounts.append(5)
		else:
			diff = int((inputNodeCount - self.classCount)/3)
			nodeCounts.append(inputNodeCount - diff)
			nodeCounts.append(inputNodeCount - 2 * diff)

		nodeCounts.append(self.classCount)
		return nodeCounts
	def getFeaturesTrain(self, features, inputNodeCount, testIndexes):
		newFeatures = np.random.random((len(features) - len(testIndexes), inputNodeCount))
		index = 0
		for i in range(len(features)):
			if i not in testIndexes:
				for j in range(inputNodeCount):
					newFeatures[index][j] = features[i][j]
				index = index + 1
		
		return newFeatures

	def getFeaturesTest(self, features, inputNodeCount, testIndexes):
		newFeatures = np.random.random((len(testIndexes), inputNodeCount))
		index = 0
		for i in range(len(features)):
			if i in testIndexes:
				for j in range(inputNodeCount):
					newFeatures[index][j] = features[i][j]
				index = index + 1
		
		return newFeatures

	def getLabelsTrain(self, labels, testIndexes):
		newLabels = np.random.random((len(labels) - len(testIndexes), self.classCount))
		index = 0
		for i in range(len(labels)):
			if i not in testIndexes:
				newLabels[index] = self.getLabel(labels[i])
				index = index + 1
		
		return newLabels

	def getLabelsTest(self, labels, testIndexes):
		newLabels = np.random.random((len(testIndexes), self.classCount))
		index = 0
		for i in range(len(labels)):
			if i in testIndexes:
				newLabels[index] = self.getLabel(labels[i])
				index = index + 1
		
		return newLabels


	def getLabel(self, label):
		if self.classCount == 3:
			if label == 0:
				return [1,0,0]
			if label == 1:
				return [1,0,0]
			if label == 2:
				return [0,1,0]
			if label == 3:
				return [0,0,1]
		elif self.classCount == 4:
			if label == 0:
				return [1,0,0,0]
			if label == 1:
				return [0,1,0,0]
			if label == 2:
				return [0,0,1,0]
			if label == 3:
				return [0,0,0,1]




