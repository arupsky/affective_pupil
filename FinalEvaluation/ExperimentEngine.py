from Helper import Helper
import json
from CNN_Re import CNN_Re
from FCNN_Re import FCNN_Re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Experiment:

	def __init__(self,sourceFolder, outputFolder,  dataType, classCount, classifierType):
		self.sourceFolder = sourceFolder
		self.outputFolder = outputFolder
		self.dataType = dataType
		self.classCount = classCount
		self.classifierType = classifierType
		self.matrixInfo = []
		self.trainInfo = []

	def getTypeStrings(self):
		if self.classCount == 2:
			return ["Neutral", "Affective"]
		elif self.classCount == 3:
			return ["Negative", "Neutral", "Positive"]
		elif self.classCount == 4:
			return ["Human Violation", "Animal Mistreatment", "Neutral", "Positive"]

	def getCnnConfiguration(self):
		modelOptions = {}
		modelOptions["epochs"] = 350
		modelOptions["batch_size"] = 64
		modelOptions["verbose"] = 0
		modelOptions["dropout"] = False

		layer1 = {
			"num_filters":10,
			"kernel_size":5,
			"dropout":.5,
			"pool_size":2
		}

		layer2 = {
			"num_filters":20,
			"kernel_size":3,
			"dropout":.5,
			"pool_size":4
		}

		modelOptions["conv_layers"] = [layer1,layer2]
		modelOptions["dense_layers"] = [10]
		return modelOptions

	def getMlpConfiguration(self):
		config = {}
		config["epochs"] = 350
		config["verbose"] = 0
		config["batch_size"] = 64
		config["layers"] = [15,6]
		return config

	def saveMatrixInfo(self):
		with open(self.outputFolder + "/report.json", "w") as file:
			json.dump(self.trainInfo, file)

	def printConfusionMatrix(self):
		for info in self.matrixInfo:
			confusionMatrixFile = info["folder"] + "/confusionMatrix.png"
			confusionMatrixNormFile = info["folder"] + "/confusionMatrixNormalized.png"
			numpyFile = info["folder"] + "/matrix.npy"
			np.save(numpyFile, info["matrix"])
			Helper.plot_confusion_matrix(info["matrix"], self.getTypeStrings(),start=info["start"], window = info["window"], fileName=confusionMatrixFile, title="")
			Helper.plot_confusion_matrix(info["matrix"], self.getTypeStrings(),start=info["start"], window = info["window"], fileName=confusionMatrixNormFile, normalize=True, title="")



	def printInfo(self):
		print(self.classifierType, "Source:", self.sourceFolder, "Out:", self.outputFolder)

	def loadTrainTestData(self):
		trainSet = []
		testSet = []

		with open(self.sourceFolder + "/train.json") as file:
			trainSet = json.load(file)

		with open(self.sourceFolder + "/test.json") as file:
			testSet = json.load(file)

		return trainSet, testSet

	def startCnnTraining(self, start, window):
		trainSet, testSet = self.loadTrainTestData()

		if len(trainSet) > 0 and len(testSet) > 0:
			# for cnn
			cnnResultFolder = Helper.createNewFolderNamed(self.outputFolder, "CNN")
			trainFeatures, trainLabels = Helper.getTrainableData(trainSet, start, window, self.classCount)
			testFeatures, testLabels = Helper.getTrainableData(testSet, start, window, self.classCount)
			cnnConfig = self.getCnnConfiguration()
			cnn = CNN_Re(trainFeatures, trainLabels, cnnConfig)
			cnnModel = cnn.trainModel(trainFeatures, trainLabels)
			_, accuracy = cnnModel.evaluate(testFeatures, testLabels, verbose=0)
			prediction = cnnModel.predict(testFeatures)
			predictedLabels = tf.argmax(prediction,axis=1)
			realLabels = tf.argmax(testLabels,axis=1)
			confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, self.classCount).numpy()
			print(confusionMatrix)
			confusionMatrixFile = cnnResultFolder + "/confusionMatrix.png"
			confusionMatrixNormFile = cnnResultFolder + "/confusionMatrixNormalized.png"
			info = {}
			info["matrix"] = confusionMatrix
			info["folder"] = cnnResultFolder
			info["start"] = start
			info["window"] = window

			trainInfo = {}
			trainInfo["experiment"] = "CNN"
			trainInfo["dataType"] = self.dataType
			trainInfo["folder"] = cnnResultFolder
			trainInfo["accuracy"] = str(accuracy)
			trainInfo["classifierType"] = self.classifierType

			self.matrixInfo.append(info)
			self.trainInfo.append(trainInfo)
			# Helper.plot_confusion_matrix(confusionMatrix, self.getTypeStrings(),start=start, window = window, fileName=confusionMatrixFile)
			# Helper.plot_confusion_matrix(confusionMatrix, self.getTypeStrings(),start=start, window = window, fileName=confusionMatrixNormFile, normalize=True)

	def startMlpTraining(self, start, window):
		trainSet, testSet = self.loadTrainTestData()

		if len(trainSet) > 0 and len(testSet) > 0:
			# for mlp
			mlpResultFolder = Helper.createNewFolderNamed(self.outputFolder, "MLP")
			trainFeatures, trainLabels = Helper.getTrainableDataFCNNReducedFeature(trainSet, start, window, self.classCount)
			testFeatures, testLabels = Helper.getTrainableDataFCNNReducedFeature(testSet, start, window, self.classCount)
			cnnConfig = self.getMlpConfiguration()
			mlp = FCNN_Re(trainFeatures, trainLabels, cnnConfig)
			mlpModel = mlp.trainModel(trainFeatures, trainLabels)
			_, accuracy = mlpModel.evaluate(testFeatures, testLabels, verbose=0)
			prediction = mlpModel.predict(testFeatures)
			predictedLabels = tf.argmax(prediction,axis=1)
			realLabels = tf.argmax(testLabels,axis=1)
			confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, self.classCount).numpy()
			print(confusionMatrix)
			confusionMatrixFile = mlpResultFolder + "/confusionMatrix.png"
			confusionMatrixNormFile = mlpResultFolder + "/confusionMatrixNormalized.png"
			info = {}
			info["matrix"] = confusionMatrix
			info["folder"] = mlpResultFolder
			info["start"] = start
			info["window"] = window

			trainInfo = {}
			trainInfo["experiment"] = "MLP"
			trainInfo["dataType"] = self.dataType
			trainInfo["folder"] = mlpResultFolder
			trainInfo["accuracy"] = str(accuracy)
			trainInfo["classifierType"] = self.classifierType

			self.matrixInfo.append(info)
			self.trainInfo.append(trainInfo)


	def startSlidingWindowTraining(self, start, window):
		trainSet, testSet = self.loadTrainTestData()

		if len(trainSet) > 0 and len(testSet) > 0:
			# for mlp
			mlpResultFolder = Helper.createNewFolderNamed(self.outputFolder, "SlidingWindow")
			trainFeatures, trainLabels = Helper.getTrainingDataSlidingWindowMlp(trainSet, 240, 10, self.classCount)
			testFeatures, testLabels = Helper.getTrainingDataSlidingWindowMlp(testSet, 240, 10, self.classCount)
			cnnConfig = self.getMlpConfiguration()
			mlp = FCNN_Re(trainFeatures, trainLabels, cnnConfig)
			mlpModel = mlp.trainModel(trainFeatures, trainLabels)
			_, accuracy = mlpModel.evaluate(testFeatures, testLabels, verbose=0)
			prediction = mlpModel.predict(testFeatures)
			predictedLabels = tf.argmax(prediction,axis=1)
			realLabels = tf.argmax(testLabels,axis=1)
			confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, self.classCount).numpy()
			print(confusionMatrix)
			confusionMatrixFile = mlpResultFolder + "/confusionMatrix.png"
			confusionMatrixNormFile = mlpResultFolder + "/confusionMatrixNormalized.png"
			info = {}
			info["matrix"] = confusionMatrix
			info["folder"] = mlpResultFolder
			info["start"] = start
			info["window"] = window
			
			trainInfo = {}
			trainInfo["experiment"] = "SlidingWindow"
			trainInfo["dataType"] = self.dataType
			trainInfo["folder"] = mlpResultFolder
			trainInfo["accuracy"] = str(accuracy)
			trainInfo["classifierType"] = self.classifierType

			self.matrixInfo.append(info)
			self.trainInfo.append(trainInfo)



class GlobalExperiment(Experiment):
	"""docstring for GlobalExperiment"""
	def __init__(self, sourceFolder, outputFolder, dataType, classCount):
		super(GlobalExperiment, self).__init__(sourceFolder, outputFolder, dataType, classCount, "Global")


class IndividualExperiment(Experiment):
	"""docstring for GlobalExperiment"""
	def __init__(self, sourceFolder, outputFolder, dataType, classCount):
		super(IndividualExperiment, self).__init__(sourceFolder, outputFolder, dataType, classCount, "Individual")
		
		
		

folderIndividual2 = "data/individual2/"
folderIndividual3 = "data/individual3/"
folderIndividual4 = "data/individual4/"
folderGlobal2 = "data/global2/"
folderGlobal3 = "data/global3/"
folderGlobal4 = "data/global4/"

globalFolders = [folderGlobal2, folderGlobal3, folderGlobal4]
individualFolders = [folderIndividual2, folderIndividual3, folderIndividual4]
foldersInRoot = ["raw", "augmented", "augmented_train"]

classes = [2,3,4]

def getDataFoldersForClass(className):
	if className == 2:
		return folderGlobal2, folderIndividual2
	elif className == 3:
		return folderGlobal3, folderIndividual3
	elif className == 4:
		return folderGlobal4, folderIndividual4

def evaluateFolder(dataFolder, resultFolder):
	# rawData = dataFolder + "raw.json"
	# augmentedData = dataFolder + "augmented.json"
	# augmentedTrainTrainData = dataFolder + "augmented_train/train.json"
	# augmentedTrainTestData = dataFolder + "augmented_train/test.json"

	# print("processing raw")
	# print(rawData)
	for folder in foldersInRoot:
		print("reading ", dataFolder, folder)


# print("create new result folder")
rootFolder = "engine_results"
resultFolder = Helper.createNewFolder(rootFolder)
experiments = []
print(resultFolder)
for className in classes:

	globalDataFolder, individualDataFolder = getDataFoldersForClass(className)

	# print("create new folder for class")
	classResultFolder = Helper.createNewFolderNamed(resultFolder, "class_" + str(className))
	# classResultFolder = resultFolder + "class_" + str(className) + "/"
	# print(classResultFolder)
	# globalResultFolder = classResultFolder + "global_classifier/"
	globalResultFolder = Helper.createNewFolderNamed(classResultFolder, "global_classifier")
	individualResultFolderRoot = Helper.createNewFolderNamed(classResultFolder, "individual_classifier")
	individualResultFolders = []
	for i in range(1,11):
		individualResultFolders.append(Helper.createNewFolderNamed(individualResultFolderRoot, str(i)))

	for folder in foldersInRoot:
		dataSpecificResult = Helper.createNewFolderNamed(globalResultFolder, folder)
		globalExperiment = GlobalExperiment(globalDataFolder + "/" + folder, dataSpecificResult, folder, className)
		experiments.append(globalExperiment)

		for i in range(len(individualResultFolders)):
			indiFolder = individualResultFolders[i]
			indiDataFolder = individualDataFolder  + str(i + 1)
			dataSpecificIndiFolder = Helper.createNewFolderNamed(indiFolder, folder)
			indiExperiment = IndividualExperiment(indiDataFolder+ "/" + folder, dataSpecificIndiFolder, folder, className)
			experiments.append(indiExperiment)


trainInfoList = []

for experiment in experiments:
	experiment.printInfo()
	experiment.startCnnTraining(0, 240)
	experiment.startMlpTraining(0, 240)
	experiment.startSlidingWindowTraining(0, 240)
	experiment.saveMatrixInfo()
	
for experiment in experiments:
	experiment.printConfusionMatrix()
	trainInfoList.extend(experiment.trainInfo)


with open(resultFolder + "/info.json", "w") as file:
	json.dump(trainInfoList, file)


plt.show()
	



	