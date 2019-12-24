import json
import numpy as np
import statistics
import matplotlib.pyplot as plt
import scipy.stats as stats


def getMeanOfData(dataset):
	result = []
	for dt in dataset:
		result.append(np.mean(dt))
	return result

def getSkewnessOfData(dataset):
	result = []
	for dt in dataset:
		result.append(stats.skew(dt))
	return result

def getKurtosisOfData(dataset):
	result = []
	for dt in dataset:
		result.append(stats.kurtosis(dt))
	return result

def getStdOfData(dataset):
	result = []
	for dt in dataset:
		result.append(np.std(dt))
	return result

def getMedianOfData(dataset):
	result = []
	for dt in dataset:
		result.append(np.median(dt))
	return result

def getModeOfData(dataset):
	result = []
	for dt in dataset:
		result.append(stats.mode(dt)[0][0])
	return result

def getMaxOfData(dataset):
	result = []
	for dt in dataset:
		result.append(max(dt))
	return result

def getMinOfData(dataset):
	result = []
	for dt in dataset:
		result.append(min(dt))
	return result

def getMaxOfData(dataset):
	result = []
	for dt in dataset:
		result.append(max(dt))
	return result

def getRangeOfData(dataset):
	result = []
	for dt in dataset:
		result.append(max(dt) - min(dt))
	return result


folder = "raw"
start = 0
window = 240

with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)


fullDataSet = []

for dt in trainSet:
	fullDataSet.append(dt)

for dt in validationSet:
	fullDataSet.append(dt)

for dt in testSet:
	fullDataSet.append(dt)


negativeData = [dt["pupilListSmoothed"][start:start + window] for dt in fullDataSet if dt["type"] == 0]
neutralData = [dt["pupilListSmoothed"][start:start + window] for dt in fullDataSet if dt["type"] == 1]
positiveData = [dt["pupilListSmoothed"][start:start + window] for dt in fullDataSet if dt["type"] == 2]

negative = getMeanOfData(negativeData)
neutral = getMeanOfData(neutralData)
positive = getMeanOfData(positiveData)

print("\nMean")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))


negative = getMedianOfData(negativeData)
neutral = getMedianOfData(neutralData)
positive = getMedianOfData(positiveData)

print("\nMedian")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))


negative = getMaxOfData(negativeData)
neutral = getMaxOfData(neutralData)
positive = getMaxOfData(positiveData)

print("\nMax")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))


negative = getMinOfData(negativeData)
neutral = getMinOfData(neutralData)
positive = getMinOfData(positiveData)

print("\nMin")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))


negative = getRangeOfData(negativeData)
neutral = getRangeOfData(neutralData)
positive = getRangeOfData(positiveData)

print("\nRange")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))



negative = getSkewnessOfData(negativeData)
neutral = getSkewnessOfData(neutralData)
positive = getSkewnessOfData(positiveData)

print("\nSkewness")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))

negative = getKurtosisOfData(negativeData)
neutral = getKurtosisOfData(neutralData)
positive = getKurtosisOfData(positiveData)

print("\nkurtosis")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))


negative = getStdOfData(negativeData)
neutral = getStdOfData(neutralData)
positive = getStdOfData(positiveData)

print("\nStd")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))

negative = getModeOfData(negativeData)
neutral = getModeOfData(neutralData)
positive = getModeOfData(positiveData)

print("\nMode")
print(stats.f_oneway(negative, neutral, positive))
print(stats.levene(negative, neutral, positive))
