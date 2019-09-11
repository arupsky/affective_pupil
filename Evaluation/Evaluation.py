from Formatter.Formatter import Formatter
from DataCollector.DataCollector import DataCollector
from UsefulRegionChecker.UsefulRegionChecker import UsefulRegionChecker
from FeatureDataProvider.FeatureDataProvider import FeatureDataProvider
from ModelProvider.ModelProvider import ModelProvider
from TrainDataProvider.TrainDataProvider import TrainDataProvider
from ModelProvider.RecurrentNeuralNetwork import RecurrentNeuralNetwork
from ModelProvider.LongShortTermMemory import LongShortTermMemory
from ModelProvider.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ModelProvider.FullyConnectedNetwork import FullyConnectedNetwork
import json
import time
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from os import listdir
from os import makedirs
from os.path import isfile, join
# style.use('ggplot')

def createNewFolder():

	folders = [int(f.split('.')[0]) for f in listdir("Reports") if not isfile(join("Reports", f))]
	if len(folders) == 0:
		outputFolderName = "0"
	else:
		outputFolderName = str(max(folders) + 1)

	makedirs("Reports/" + outputFolderName)
	return "Reports/" + outputFolderName + "/"

def plot_confusion_matrix(confusionMatrix, classes,normalize=False,title=None,cmap=plt.cm.Blues,start=0,window=120,fileName="",modelName=""):
		
	if not title:
			if normalize:
					title = 'Normalized confusion matrix'
			else:
					title = 'Confusion matrix '+modelName+': start - ' + str(start) + ", window - " + str(window)

	cm = confusionMatrix

	if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
	else:
			print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
				 yticks=np.arange(cm.shape[0]),
				 # ... and label them with the respective list entries
				 xticklabels=classes, yticklabels=classes,
				 title=title,
				 ylabel='True label',
				 xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
					 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
					ax.text(j, i, format(cm[i, j], fmt),
									ha="center", va="center",
									color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	if fileName != "":
		plt.savefig(fileName, dpi=400)
	return ax



debugInfo = []
starts = [0, 12]
windows = [90, 120, 180, 240, 270]
# windows = [120]

# variables for 3d plot ##############

x3 = []
y3 = []
z3 = np.zeros(len(windows) * len(starts))
dx = np.ones(len(windows) * len(starts))
dy = np.ones(len(windows) * len(starts))
dzCNN = []
dzFCNN = []
# ------------------------------------


outputFolderPath = createNewFolder()
dataCollector = DataCollector("../../data/")
dataJson = dataCollector.loadFolder()
formatter = Formatter()
formattedData = formatter.process(dataJson)
trainDataProvider = TrainDataProvider(formattedData)


for window in windows:
	for start in starts:
		if start + window > formatter.interestingZone:
			continue

		print("------------------------------------------------------")
		print("Start ", (start), ", window ", window)
		
		# cnn training
		trainFeatures, trainLabels = trainDataProvider.getTrainableData(start, window)
		cnn = ConvolutionalNeuralNetwork(trainFeatures, trainLabels, epochs=50)
		
		# fcnn training
		trainFeaturesFCNN, trainLabelsFCNN = trainDataProvider.getTrainableDataFCNN(start, window)
		fcnn = FullyConnectedNetwork(trainFeaturesFCNN, trainLabelsFCNN, epochs=50)

		# cnn confusion matrix
		confusionMatrixFileName = outputFolderPath+"ConfusionMatrix-CNN-"+str(start)+"-"+str(window)+".png"
		plot_confusion_matrix(cnn.confusionMatrices[0], ["negative", "neutral", "positive"], start=start, window=window, fileName=confusionMatrixFileName, modelName="CNN")

		# fcnn confusion matrix
		confusionMatrixFileName = outputFolderPath+"ConfusionMatrix-FCNN-"+str(start)+"-"+str(window)+".png"
		plot_confusion_matrix(fcnn.confusionMatrices[0], ["negative", "neutral", "positive"], start=start, window=window, fileName=confusionMatrixFileName, modelName="FCNN")
		
		# report data
		debug = {}
		debug["start"] = start
		debug["window"] = window
		debug["cnn_efficiency"] = cnn.efficiency
		debug["cnn_error"] = cnn.error
		debug["fcnn_efficiency"] = fcnn.efficiency
		debug["fcnn_error"] = fcnn.error
		debugInfo.append(debug)

		# efficiency 3d graph data
		x3.append(window)
		y3.append(start)
		dzCNN.append(cnn.efficiency)
		dzFCNN.append(fcnn.efficiency)

with open(outputFolderPath+"report-"+str(time.time())+".json", "w") as file:
	json.dump(debugInfo, file)

# CNN efficiency 3d plot ###################
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.bar3d(x3, y3, z3, dx, dy, dzCNN)
ax1.set_xlabel('window size')
ax1.set_ylabel('start index')
ax1.set_zlabel('efficiency (%)')
plt.savefig(outputFolderPath+"start-window-efficiency(cnn).png",dpi=400)

# ------------------------------------

# FCNN efficiency 3d plot ###################
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.bar3d(x3, y3, z3, dx, dy, dzFCNN)
ax1.set_xlabel('window size')
ax1.set_ylabel('start index')
ax1.set_zlabel('efficiency (%)')
plt.savefig(outputFolderPath+"start-window-efficiency(fcnn).png",dpi=400)

# ------------------------------------

plt.show()


# ---------------- /Tensorflow Model-------------

# window = 180

# for start in range(int((formatter.interestingZone - window)/10)):
# 	print("------------------------------------------------------")
# 	print("Start ", (start * 10), ", window ", window)
# 	trainFeatures, trainLabels = trainDataProvider.getTrainableData(start * 10, window)
# 	# rnn = RecurrentNeuralNetwork(trainFeatures, trainLabels)
# 	# lstm = LongShortTermMemory(trainFeatures, trainLabels)
# 	cnn = ConvolutionalNeuralNetwork(trainFeatures, trainLabels)
# 	# efficiency = cnn.efficiency
	
# 	debug = {}
# 	debug["start"] = start
# 	debug["window"] = window
# 	debug["cnn_efficiency"] = cnn.efficiency
# 	debug["cnn_error"] = cnn.error
# 	debugInfo.append(debug)

# 		# debug["rnn"] = rnn.getEfficiency()
# 		# debug["lstm"] = lstm.getEfficiency()
# 		# debug["cnn"] = cnn.getEfficiency()

# 		debugInfo.append(debug)
# 		print("--------------------------------------------------")
# 		print(debug)
# 		rnn.printEvaluation()
# 		lstm.printEvaluation()
# 		cnn.printEvaluation()

# print("Total experiments : ", len(debugInfo))

# with open("Reports/report"+str(window)+".json", "w") as file:
# 	json.dump(debugInfo, file)
