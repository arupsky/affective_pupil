import json
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
from os import makedirs
from os import path


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

def getBspline(data, length=100, degree=3):
	controlPoints = []
	for i in range(len(data)):
		controlPoints.append([i,data[i]])
	cv = np.array(controlPoints)
	p = bspline(cv,n=length,degree=degree,periodic=False)
	x,y = p.T
	return y

def getSyntheticData(pupilList, start = 0, end = 150):
	window_size = int((end - start) * .1)
	window_start = random.choice(range(start, end - window_size))
	scale_factor = .2 + .3 * random.random()
	if random.random() < .5:
		target_window = window_size + int(window_size * scale_factor)
	else:
		target_window = window_size - int(window_size * scale_factor)

	# fig = plt.figure()
	# ax1 = fig.add_subplot(311)
	# ax1.plot(pupilList, label='source')
	# ax1.axvline(x=window_start, color='red', linestyle='--')
	# ax1.axvline(x=(window_start + window_size), color='red', linestyle='--')
	# plt.legend()

	# print("window_size", window_size)
	# print("window_start", window_start)
	# print("target_window", target_window)

	augmentPart = getBspline(pupilList[window_start:window_start + window_size], length=target_window, degree=3)

	newPupilList = pupilList[:window_start]
	newPupilList.extend(augmentPart)
	newPupilList.extend(pupilList[window_start + window_size:])


	# ax2 = fig.add_subplot(312)
	# ax2.plot(newPupilList, label='window warping')
	# ax2.axvline(x=window_start, color='red', linestyle='--')
	# ax2.axvline(x=(window_start + target_window), color='red', linestyle='--')
	# plt.legend()

	smoothed = getBspline(newPupilList, length=len(newPupilList), degree=3)

	# ax3 = fig.add_subplot(313)
	# ax3.plot(smoothed, label='final')
	# plt.legend()


	return list(smoothed)

def getAugmented(trial):
	augmented = {}
	augmented["type"] = trial["type"]
	augmented["pupilListSmoothed"] = getSyntheticData(trial["pupilListSmoothed"])
	return augmented


def getType2Version(dataset):
	dt2 = []
	for data in dataset:
		temp = {}
		temp["pupilListSmoothed"] = data["pupilListSmoothed"]
		if data["type"] == 1:
			temp["type"] = 0
		else:
			temp["type"] = 1

		dt2.append(temp)

	type0 = [dt for dt in dt2 if dt["type"] == 0]
	for data in type0:
		augmented = getAugmented(data)
		dt2.append(augmented)

	return dt2

def getType3Version(dataset):
	dt3 = []
	for data in dataset:
		temp = {}
		temp["pupilListSmoothed"] = data["pupilListSmoothed"]
		temp["type"] = int(data["type"])
		dt3.append(temp)

	return dt3

def getType4Version(dataset):
	dt4 = []
	for data in dataset:
		temp = {}
		temp["pupilListSmoothed"] = data["pupilListSmoothed"]
		
		if int(data["type"]) > 0:
			temp["type"] = int(data["type"]) + 1
		else:
			if "H" in data["imageName"]:
				temp["type"] = 0
			else:
				temp["type"] = 1

		dt4.append(temp)

	type0 = [dt for dt in dt4 if dt["type"] == 0]
	type1 = [dt for dt in dt4 if dt["type"] == 1]

	for data in type0:
		augmented = getAugmented(data)
		dt4.append(augmented)

	for data in type1:
		augmented = getAugmented(data)
		dt4.append(augmented)

	return dt4

folderIndividual2 = "data/individual2/"
folderIndividual3 = "data/individual3/"
folderIndividual4 = "data/individual4/"
folderGlobal2 = "data/global2/"
folderGlobal3 = "data/global3/"
folderGlobal4 = "data/global4/"

globalFolders = [folderGlobal2, folderGlobal3, folderGlobal4]
individualFolders = [folderIndividual2, folderIndividual3, folderIndividual4]



#### Generate Raw File
def generateRawFiles():
	folderSource = "data/source/"

	fileNames = list(range(1,11))
	fileNames = [str(i) for i in fileNames]

	raw3File = folderGlobal3 + "raw.json"
	raw2File = folderGlobal2 + "raw.json"
	raw4File = folderGlobal4 + "raw.json"

	data3 = []
	data2 = []
	data4 = []
	for fileName in fileNames:
		with open(folderSource + fileName + ".json") as file:
			dt = json.load(file)
			dt3 = getType3Version(dt)
			dt2 = getType2Version(dt)
			dt4 = getType4Version(dt)

			data3.extend(dt3)
			data2.extend(dt2)
			data4.extend(dt4)

			with open(folderIndividual3 + "/" + fileName +"/raw.json", "w") as indiFile:
				json.dump(dt3, indiFile)

			with open(folderIndividual2 + "/" + fileName +"/raw.json", "w") as indiFile:
				json.dump(dt2, indiFile)

			with open(folderIndividual4 + "/" + fileName +"/raw.json", "w") as indiFile:
				json.dump(dt4, indiFile)



			

		with open(raw3File, "w") as file:
			json.dump(data3, file)

		with open(raw2File, "w") as file:
			json.dump(data2, file)

		with open(raw4File, "w") as file:
			json.dump(data4, file)

### Raw File Generate complete

def getClasses(dataset):
	classList = [int(dt["type"]) for dt in dataset]
	classes = set(classList)
	counts = []
	for className in classes:
		data = [dt for dt in dataset if int(dt["type"]) == className]
		counts.append(len(data))
	return list(classes), counts

def getAugmentedList(dataset):
	newList = []
	for data in dataset:
		newList.append(getAugmented(data))
		newList.append(getAugmented(data))
	return newList

def generateAugmented(folder, processType = ""):
	with open(folder + "raw.json") as file:
		print("----------",processType, folder, "-------------")
		data = json.load(file)
		print("old length",len(data),"old class",getClasses(data))
		newAugmented = getAugmentedList(data)
		newAugmented.extend(data)
		random.shuffle(newAugmented)
		print("new length",len(newAugmented),"old class",getClasses(newAugmented))
		augmentedFileName = folder + "augmented.json"
		with open(augmentedFileName, "w") as augmentedFile:
			json.dump(newAugmented, augmentedFile)

def generateAugmentedTrain(folder):
	rawFile = folder + "raw.json"
	augmentedTrainFile = folder + "augmented_train/train.json"
	rawTestFile = folder + "augmented_train/test.json"
	with open(rawFile) as file:
		rawData = json.load(file)
		classes, counts = getClasses(rawData)
		avgCount = int(sum(counts) / len(counts))
		testCount = int(avgCount * .4)

		testSet = []
		trainSet = []

		for className in classes:
			classData = [dt for dt in rawData if int(dt["type"]) == className]
			testTemp = random.sample(classData, testCount)
			trainTemp = [dt for dt in classData if dt not in testTemp]
			trainSet.extend(trainTemp)
			testSet.extend(testTemp)

		augmentedTrain = getAugmentedList(trainSet)
		trainSet.extend(augmentedTrain)

		with open(augmentedTrainFile, "w") as trainFile:
			json.dump(trainSet, trainFile)

		with open(rawTestFile, "w") as testFile:
			json.dump(testSet, testFile)

		print("Train set ", getClasses(trainSet), "Test set ", getClasses(testSet))


def processGlobalFiles():
	print("Augmenting global files")
	for folder in globalFolders:
		generateAugmented(folder)
		generateAugmentedTrain(folder)

def processIndividualFiles():
	print("\nAugmenting individual files")
	for folderName in individualFolders:
		for i in range(1,11):
			folder = folderName + str(i) + "/"
			generateAugmented(folder, "individual")
			generateAugmentedTrain(folder)

def convertToTrainTest(folder, fileName):
	targetFolder = folder + fileName + "/"

	if not path.exists(folder + fileName):
		print("creating ", (folder + fileName))
		makedirs(folder + fileName)

	print("creating folder ", (folder + fileName))
	sourceFile = folder + fileName + ".json"
	trainFileName = targetFolder + "train.json"
	testFileName = targetFolder + "test.json"
	print("reading file ", sourceFile)
	print("generating", trainFileName)
	print("generating", testFileName)

	with open(sourceFile) as file:
		rawData = json.load(file)
		classes, counts = getClasses(rawData)
		avgCount = int(sum(counts) / len(counts))
		testCount = int(avgCount * .4)

		testSet = []
		trainSet = []

		for className in classes:
			classData = [dt for dt in rawData if int(dt["type"]) == className]
			testTemp = random.sample(classData, testCount)
			trainTemp = [dt for dt in classData if dt not in testTemp]
			trainSet.extend(trainTemp)
			testSet.extend(testTemp)

		augmentedTrain = getAugmentedList(trainSet)
		trainSet.extend(augmentedTrain)

		with open(trainFileName, "w") as trainFile:
			json.dump(trainSet, trainFile)

		with open(testFileName, "w") as testFile:
			json.dump(testSet, testFile)

		print("Train set ", getClasses(trainSet), "Test set ", getClasses(testSet))



# generateRawFiles()
# processGlobalFiles()
# processIndividualFiles()

for folder in globalFolders:
	convertToTrainTest(folder, "raw")
	convertToTrainTest(folder, "augmented")

for folderRoot in individualFolders:
	for i in range(1,11):
		folder = folderRoot + str(i) + "/"
		convertToTrainTest(folder, "raw")
		convertToTrainTest(folder, "augmented")

