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
import scipy.interpolate as si
from tsvparser import TsvParser
import random
from Helper import Helper
import math



# style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"

config = {}
config["dummy"] = False
config["participant"] = -1
config["do_cnn"] = True
config["do_fcnn"] = True
config["generate_confusion_matrix"] = True
config["generate_mean_graph"] = True

config["generate_preprocess_graph"] = True   # Fertig
config["preprocess_graph_count_per_file"] = 2   # Fertig

config["generate_baseline_normalization_graph"] = True
config["debug_data_collect"] = False   # Fertig


def printBlockStartSeperator():
	print("\n----------------------------------------------------------------------------")

def printBlockEndSeperator():
	print("-----------------------------------\n")

def createNewFolder():

	folders = [int(f.split('.')[0]) for f in listdir("Reports") if not isfile(join("Reports", f))]
	if len(folders) == 0:
		outputFolderName = "0"
	else:
		outputFolderName = str(max(folders) + 1)
	print("Creating folder for output...")
	makedirs("Reports/" + outputFolderName)
	print("Folder creation successful. Output folder name : ", ("Reports/" + outputFolderName + "/"))
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

def plotRawData(formattedData, countPerType):
	negativeData = [dt for dt in formattedData if dt['type'] <= 1]
	neutralData = [dt for dt in formattedData if dt['type'] == 2]
	positiveData = [dt for dt in formattedData if dt['type'] == 3]

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	plt.xlabel("time frame")
	plt.ylabel("pupil diameter (mm)")
	

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	plt.xlabel("time frame")
	plt.ylabel("pupil diameter (mm)")

	for i in range(countPerType):

		if i == 0:
			ax1.plot(negativeData[i]['pupilList'], 'r', label="Negative")
			ax1.plot(neutralData[i]['pupilList'], 'b', label="Neutral")
			ax1.plot(positiveData[i]['pupilList'], 'g', label="Positive")
		else:
			ax1.plot(negativeData[i]['pupilList'], 'r')
			ax1.plot(neutralData[i]['pupilList'], 'b')
			ax1.plot(positiveData[i]['pupilList'], 'g')


		tempNeg = [dt - negativeData[i]['baselineMean'] for dt in negativeData[i]['pupilList']]
		tempNeu = [dt - neutralData[i]['baselineMean'] for dt in neutralData[i]['pupilList']]
		tempPos = [dt - positiveData[i]['baselineMean'] for dt in positiveData[i]['pupilList']]

		if i == 0:
			ax2.plot(tempNeg, 'r', label="Negative")
			ax2.plot(tempNeu, 'b', label="Neutral")
			ax2.plot(tempPos, 'g', label="Positive")
		else:
			ax2.plot(tempNeg, 'r')
			ax2.plot(tempNeu, 'b')
			ax2.plot(tempPos, 'g')

	# fig1.xlabel("time frame")
	# fig1.ylabel("mean pupil diameter change (mm)")
	ax1.legend()

	# fig2.xlabel("time frame")
	# fig2.ylabel("mean pupil diameter change (mm)")
	ax2.legend()

def smooth(dataSet, window):
	result = []
	hand = int((window-1)/2)
	for i in range(len(dataSet)):
		if i < hand:
			mean = sum(dataSet[:i+hand+1])/(i+hand+1) 
		else:
			mean = sum(dataSet[i-hand:i+hand+1])/window
		result.append(mean)
	return result

def bspline(cv, n=100, degree=3, periodic=False):
	""" Calculate n samples on a bspline

		cv :      Array ov control vertices
		n  :      Number of samples to return
		degree:   Curve degree
		periodic: True - Curve is closed
				  False - Curve is open
	"""

	# If periodic, extend the point array by count+degree+1
	cv = np.asarray(cv)
	count = len(cv)

	if periodic:
		factor, fraction = divmod(count+degree+1, count)
		cv = np.concatenate((cv,) * factor + (cv[:fraction],))
		count = len(cv)
		degree = np.clip(degree,1,degree)

	# If opened, prevent degree from exceeding count-1
	else:
		degree = np.clip(degree,1,count-1)


	# Calculate knot vector
	kv = None
	if periodic:
		kv = np.arange(0-degree,count+degree+degree-1)
	else:
		kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

	# Calculate query range
	u = np.linspace(periodic,(count-degree),n)


	# Calculate result
	return np.array(si.splev(u, (kv,cv.T,degree))).T

def spline(pupilList):
	smoothed = smooth(pupilList, 5)
	
	plt.figure()
	plt.plot(pupilList, 'y', label='raw')
	plt.plot(smoothed, label='smoothed')
	plt.legend()
	
	controlPoints = []
	for i in range(len(pupilList)):
		controlPoints.append([i,pupilList[i]])
	cv = np.array(controlPoints)
	p = bspline(cv,n=len(pupilList) * 2,degree=16,periodic=False)
	x,y = p.T
	plt.figure()
	plt.plot(pupilList, 'y', label='raw')
	plt.plot(x,y,'k-',label='b-spline degree 16')
	plt.legend()

	controlPoints = []
	for i in range(len(smoothed)):
		controlPoints.append([i,smoothed[i]])
	cv = np.array(controlPoints)
	p = bspline(cv,n=len(pupilList),degree=4,periodic=False)
	x,y = p.T
	plt.figure()
	plt.plot(pupilList, 'y', label='raw')
	plt.plot(y, label='smooth + b-spline degree 4')
	plt.legend()

	controlPoints = []
	base = 90
	segmentLength = int(300/base)
	segment = 0
	while segment * segmentLength < len(pupilList):
		controlPoints.append([segment, pupilList[segment * segmentLength]])
		segment = segment + 1
	if (segment - 1) * segmentLength != len(pupilList) - 1:
		controlPoints.append([segment, pupilList[len(pupilList) - 1]])

	cv = np.array(controlPoints)
	p = bspline(cv,n=len(pupilList),degree=4,periodic=False)
	x,y = p.T
	plt.figure()
	plt.plot(pupilList, 'y', label='raw')
	plt.plot(y, label='b-spline degree 4 base %s'%base)
	plt.legend()


def slope(dataList):
	result = []
	for i in range(len(dataList)):
		if i == 0:
			result.append(0)
			continue

		result.append(abs(dataList[i] - dataList[i-1]))
	return result

def getMeanLine(data):
	count = len(data)
	length = len(data[0])
	mean_data = []
	for i in range(length):
		temp = []
		for sample in data:
			temp.append(sample[i])
		mean_data.append(sum(temp)/len(temp))

	return mean_data

def anomalyDetection(formattedData):
	# dt = formattedData[0]['pupilList']
	# dt = [d - formattedData[0]['baselineMean'] for d in dt]
	# smoothed = np.array(smooth(dt, 5))
	# upper = np.maximum(smoothed * 1.025, smoothed + .2)

	# indlist = [70,61,56,48,41,37,31,23,14,13,10,8,1]
	indlist = [70]
	# negativeData = [dt for dt in formattedData]
	counts = []
	start = 1
	for data in formattedData:
		# counts.append(sum([1 for dt in data['pupilList'] if dt <= 1.5]))
		# counts.append(max(data['pupilList']))


		dt = np.array(data['pupilList']) - data['baselineMean']
		smoothed = smooth(dt, 5)
		data['pupilListSmoothed'] = smoothed
		
		if start == 70:
			spline(dt)

		
		if start in indlist:
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			ax1.plot(dt,'y')
			ax1.plot(smoothed, 'b')
			
		# plt.hist(slopeDt, 20, facecolor='blue', alpha=0.5)
		
		start = start + 1

	return formattedData

	# print(counts)

	# dt = formattedData[0]['pupilList']
	# dt = [d - formattedData[0]['baselineMean'] for d in dt]
	# slopeDt = slope([dt],1)
	# fig = plt.figure()
	# plt.plot(dt,'r')
	# plt.plot(smoothed,'g')
	# plt.plot(upper, 'b')
	# plt.hist(slopeDt, 20, facecolor='blue', alpha=0.5)


def getDummyDataset():

	data = []
	for i in range(200):
		temp = {}
		temp["baselineList"] = [0 for x in range(300)]
		temp["pupilListSmoothed"] = [np.sin(math.radians(x)) * random.randint(1,3) for x in range(300)]
		temp["type"] = 0
		data.append(temp)

	for i in range(200):
		temp = {}
		temp["baselineList"] = [0 for x in range(300)]
		temp["pupilListSmoothed"] = [np.cos(math.radians(x)) * random.randint(1,3) for x in range(300)]
		temp["type"] = 1
		data.append(temp)

	for i in range(200):
		val = random.randint(1,3)
		temp = {}
		temp["baselineList"] = [0 for x in range(300)]
		temp["pupilListSmoothed"] = [val for x in range(300)]
		temp["type"] = 2
		data.append(temp)

	return data



debugInfo = []
starts = [45, 60]
# windows = [90, 120, 180, 240, 270]
windows = [60, 90]
ran_state = int(np.random.random_sample() * 100)
# ran_state = 91
# ran_state = 46
n_epochs = 300



# variables for 3d plot ##############

x3 = []
y3 = []
z3 = np.zeros(len(windows) * len(starts))
dx = np.ones(len(windows) * len(starts))
dy = np.ones(len(windows) * len(starts))
dzCNN = []
dzFCNN = []
# ------------------------------------

printBlockStartSeperator()
print("Experiment started")
print("Configuration")
print(config)
printBlockEndSeperator()

printBlockStartSeperator()
outputFolderPath = createNewFolder()
printBlockEndSeperator()

# dataCollector = DataCollector("../../data/")
# dataJson = dataCollector.loadFolder()

printBlockStartSeperator()
print("Data loading started\n")
tsvParser = TsvParser("../tobi_data/", globalConfig = config)
tsvData = tsvParser.loadData()
print("Data loading complete")
printBlockEndSeperator()

printBlockStartSeperator()
print("Data processing started")
tsvData = tsvParser.processData(tsvData)
print("Data processing complete")
printBlockEndSeperator()

if config["dummy"]:
	tsvData = getDummyDataset()

# TsvParser.generateSampleFigures(tsvData)
printBlockStartSeperator()
print("Random state : ", ran_state)
print("Total samples", len(tsvData))
print("Negative ", sum([1 for x in tsvData if x["type"] == 0]))
print("Neutral ", sum([1 for x in tsvData if x["type"] == 1]))
print("Positive ", sum([1 for x in tsvData if x["type"] == 2]))
printBlockEndSeperator()

printBlockStartSeperator()
print("Baseline normalization started")
formatter = Formatter()
# formattedData = formatter.process(dataJson)
formattedData = formatter.process(tsvData)
print("Baseline normalization complete")
printBlockEndSeperator()

if config["generate_mean_graph"]:
	printBlockStartSeperator()
	print("Creating Mean Graph")
	negative_data = [x["pupilListSmoothed"] for x in tsvData if x["type"] == 0]
	neutral_data = [x["pupilListSmoothed"] for x in tsvData if x["type"] == 1]
	positive_data = [x["pupilListSmoothed"] for x in tsvData if x["type"] == 2]

	mean_negative = Helper.smooth(getMeanLine(negative_data), 1)
	mean_neutral = Helper.smooth(getMeanLine(neutral_data), 1)
	mean_positive = Helper.smooth(getMeanLine(positive_data), 1)

	plt.figure()
	plt.plot(mean_negative, 'r', label='negative')
	plt.plot(mean_neutral, 'b', label='neutral')
	plt.plot(mean_positive, 'g', label='positive')
	plt.legend()
	printBlockEndSeperator()


# for i in range(20):
# 	plt.figure()
# 	plt.plot(formattedData[i]["pupilListSmoothed"])
# 	plt.plot(tsvData[i]["pupilList"])

# formattedData = anomalyDetection(formattedData)
trainDataProvider = TrainDataProvider(formattedData)


# plotRawData(formattedData, 30)



for window in windows:
	for start in starts:
		if start + window > formatter.interestingZone:
			continue

		print("------------------------------------------------------")
		print("Start ", (start), ", window ", window)
		
		# cnn training
		if config["do_cnn"]:
			trainFeatures, trainLabels = trainDataProvider.getTrainableData(start, window)
			cnn = ConvolutionalNeuralNetwork(trainFeatures, trainLabels, epochs=n_epochs, random_state = ran_state)
			# cnn confusion matrix
			confusionMatrixFileName = outputFolderPath+"ConfusionMatrix-CNN-"+str(start)+"-"+str(window)+".png"
			plot_confusion_matrix(cnn.confusionMatrices[0], ["negative", "neutral", "positive"], start=start, window=window, fileName=confusionMatrixFileName, modelName="CNN")

		
		# fcnn training
		if config["do_fcnn"]:
			trainFeaturesFCNN, trainLabelsFCNN = trainDataProvider.getTrainableDataFCNN(start, window)
			fcnn = FullyConnectedNetwork(trainFeaturesFCNN, trainLabelsFCNN, epochs=n_epochs, random_state = ran_state)
			# fcnn confusion matrix
			confusionMatrixFileName = outputFolderPath+"ConfusionMatrix-FCNN-"+str(start)+"-"+str(window)+".png"
			plot_confusion_matrix(fcnn.confusionMatrices[0], ["negative", "neutral", "positive"], start=start, window=window, fileName=confusionMatrixFileName, modelName="FCNN")
		
		# report data
		debug = {}
		debug["state"] = ran_state
		debug["start"] = start
		debug["window"] = window
		if config["do_cnn"]:
			debug["cnn_efficiency"] = cnn.efficiency
			debug["cnn_error"] = cnn.error
		if config["do_fcnn"]:
			debug["fcnn_efficiency"] = fcnn.efficiency
			debug["fcnn_error"] = fcnn.error
		debugInfo.append(debug)

		# efficiency 3d graph data
		x3.append(window)
		y3.append(start)
		if config["do_cnn"]:
			dzCNN.append(cnn.efficiency)
		if config["do_fcnn"]:
			dzFCNN.append(fcnn.efficiency)

with open(outputFolderPath+"report-"+str(time.time())+".json", "w") as file:
	json.dump(debugInfo, file)

# CNN efficiency 3d plot ###################
if config["do_cnn"]:
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.bar3d(x3, y3, z3, dx, dy, dzCNN)
	ax1.set_xlabel('window size')
	ax1.set_ylabel('start index')
	ax1.set_zlabel('efficiency (%)')
	plt.savefig(outputFolderPath+"start-window-efficiency(cnn).png",dpi=400)

# ------------------------------------

# FCNN efficiency 3d plot ###################
if config["do_fcnn"]:
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.bar3d(x3, y3, z3, dx, dy, dzFCNN)
	ax1.set_xlabel('window size')
	ax1.set_ylabel('start index')
	ax1.set_zlabel('efficiency (%)')
	plt.savefig(outputFolderPath+"start-window-efficiency(fcnn).png",dpi=400)

# ------------------------------------



plt.show()

