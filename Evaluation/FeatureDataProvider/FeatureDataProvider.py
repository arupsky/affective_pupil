from os import listdir
from os import makedirs
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import math  

from scipy.fftpack import fft
import numpy as np



class FeatureDataProvider(object):
	"""docstring for FeatureDataProvider"""
	def __init__(self, fileName=""):
		super(FeatureDataProvider, self).__init__()
		self.fileName = fileName

		self.inputFolder = "Formatter/Outputs/"
		self.outputFolderRoot = "FeatureDataProvider/Outputs/"
		self.data = []
		self.sampling_rate = 1000
		self.createNewFolder()
		self.loadInputData()
		self.process()
		plt.show()

	def createNewFolder(self):
		folders = [int(f.split('.')[0]) for f in listdir(self.outputFolderRoot) if not isfile(join(self.inputFolder, f))]
		if len(folders) == 0:
			self.outputFolderName = "0"
		else:
			self.outputFolderName = str(max(folders) + 1)

		makedirs(self.outputFolderRoot + self.outputFolderName)

		print(self.outputFolderName)


	def loadInputData(self):
		dataFiles = [int(f.split('.')[0]) for f in listdir(self.inputFolder) if (isfile(join(self.inputFolder, f)) and ".json" in f)]
		if len(dataFiles) == 0:
			print("No input file found")
			return
		latestFile = str(max(dataFiles)) + ".json"
		
		with open(self.inputFolder + latestFile) as file:
			self.data = json.load(file)

		print("File loaded ", (self.inputFolder + latestFile), ", count ", len(self.data))


	def saveOutputFile(self, saveData, fileName):
		# print(saveData)
		# return

		with open(join((self.outputFolderRoot + self.outputFolderName),fileName + ".json"), "w") as file:
			json.dump(saveData, file)
		# print(fileName, ", ", saveData)
		plt.figure(fileName)
		for dt in saveData["features"]:
			plt.plot(dt)

	def getMeanSdVariance(self, dataSet):

		mean = sum(dataSet) / len(dataSet)
		sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in dataSet]) / len(dataSet))
		variance = math.sqrt((sum([(x * x) for x in dataSet]) - ((sum([x for x in dataSet]) * sum([x for x in dataSet]))/len(dataSet))) / (len(dataSet) - 1))

		return [mean,sd,variance]

	def process(self):
		print("Starting process")
		# create new folder for output files

		self.saveOutputFile(self.rawDataAtSecond(0), "raw_s_1") #60
		self.saveOutputFile(self.rawDataAtSecond(1), "raw_s_2") #60
		self.saveOutputFile(self.rawDataAtSecond(2), "raw_s_3") #60

		self.saveOutputFile(self.normalizedDataAtSecond(0), "norm_s_1") #60
		self.saveOutputFile(self.normalizedDataAtSecond(1), "norm_s_2") #60
		self.saveOutputFile(self.normalizedDataAtSecond(2), "norm_s_3") #60
		self.saveOutputFile(self.normalizedFull(), "norm")

		self.saveOutputFile(self.commonFeatureAtSecond(0), "common_s_1") # mean, sd, variance
		self.saveOutputFile(self.commonFeatureAtSecond(1), "common_s_2") # mean, sd, variance
		self.saveOutputFile(self.commonFeatureAtSecond(2), "common_s_3") # mean, sd, variance
		self.saveOutputFile(self.commonFeaturesFull(), "common") # mean, sd, variance

		self.saveOutputFile(self.fftAtSecond(0), "fft_s_1")
		self.saveOutputFile(self.fftAtSecond(1), "fft_s_2")
		self.saveOutputFile(self.fftAtSecond(2), "fft_s_3")
		self.saveOutputFile(self.fftFull(), "fft")

		self.saveOutputFile(self.movingWindowFFTAtSecond(0, 8), "window_fft_s_1_8")
		self.saveOutputFile(self.movingWindowFFTAtSecond(0, 16), "window_fft_s_1_16")
		self.saveOutputFile(self.movingWindowFFTAtSecond(0, 32), "window_fft_s_1_32")
		self.saveOutputFile(self.movingWindowFFTAtSecond(1, 8), "window_fft_s_2_8")
		self.saveOutputFile(self.movingWindowFFTAtSecond(1, 16), "window_fft_s_2_16")
		self.saveOutputFile(self.movingWindowFFTAtSecond(1, 32), "window_fft_s_2_32")
		self.saveOutputFile(self.movingWindowFFTAtSecond(2, 8), "window_fft_s_3_8")
		self.saveOutputFile(self.movingWindowFFTAtSecond(2, 16), "window_fft_s_3_16")
		self.saveOutputFile(self.movingWindowFFTAtSecond(2, 32), "window_fft_s_3_32")


	def rawDataAtSecond(self, second):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			feature.append(instance["pupilList"][second*60:(second + 1) * 60])

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def normalizedDataAtSecond(self, second):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [(x / instance["baselineMean"]) for x in instance["pupilList"][second*60:(second + 1) * 60]]
			feature.append(temp)

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def commonFeatureAtSecond(self, second):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [(x) for x in instance["pupilList"][second*60:(second + 1) * 60]]
			# mean = sum(temp) / len(temp)
			# sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in temp]) / len(temp))
			# variance = math.sqrt((sum([(x * x) for x in temp]) - ((sum([x for x in temp]) * sum([x for x in temp]))/len(temp))) / (len(temp) - 1))

			feature.append(self.getMeanSdVariance(temp))
			feature[len(feature) - 1].append(instance["baselineMean"])

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def fftAtSecond(self, second):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [float(x) for x in instance["pupilList"][second*60:(second + 1) * 60]]
			peakFFT = self.calculatePeakFrequency(temp, len(temp))
			# peakFFT = self.fft.calculatePeakFrequency(temp)
			# print(peakFFT)
			# mean = sum(temp) / len(temp)
			# sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in temp]) / len(temp))
			# variance = math.sqrt((sum([(x * x) for x in temp]) - ((sum([x for x in temp]) * sum([x for x in temp]))/len(temp))) / (len(temp) - 1))
			peakFFT.append(instance["baselineMean"])
			feature.append(peakFFT)
			# break

		output = {}
		output["features"] = feature
		output["labels"] = labels
		return output

	def movingWindowFFTAtSecond(self, second, frameWidth):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [float(x) for x in instance["pupilList"][second*60:(second + 1) * 60]]
			# print(temp)

			start = 0
			dataSeries = []
			while start < len(temp) - frameWidth:
				peakMagnitude = self.calculatePeakMagnitude(temp[start:start+frameWidth])
				dataSeries.append(peakMagnitude)
				start = start + 1
				pass

			# mean = sum(temp) / len(temp)
			# sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in temp]) / len(temp))
			# variance = math.sqrt((sum([(x * x) for x in temp]) - ((sum([x for x in temp]) * sum([x for x in temp]))/len(temp))) / (len(temp) - 1))
			dataSeries.append(instance["baselineMean"])
			feature.append(dataSeries)
			# break

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}
		pass

	def normalizedFull(self):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [(x / instance["baselineMean"]) for x in instance["pupilList"]]
			feature.append(temp)

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def commonFeaturesFull(self):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			feature.append(self.getMeanSdVariance(instance["pupilList"]))
			feature[len(feature) - 1].append(instance["baselineMean"])

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def calculatePeakMagnitude(self, data):
		windowSize = len(data)
		N = windowSize;
		T = 1/self.sampling_rate # inverse of the sampling rate
		x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
		yr = fft(data)
		# print(yr[0].imag)
		y = 2/N * np.abs(yr[0:np.int(N/2)])
		maxY = max(y)
		return maxY

	def calculatePeakFrequency(self, data, windowSize):
		# print(windowSize , " ", data)
		# print(data)
		# return 0
		windowSize = len(data)
		N = windowSize;
		T = 1/self.sampling_rate # inverse of the sampling rate
		x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
		yr = fft(data)
		# print(yr[0].imag)
		y = 2/N * np.abs(yr[0:np.int(N/2)])
		topMagnitudes = -np.unique(np.sort(-y))[:3]
		indexes = [int(np.where(y == [val])[0][0]) for val in topMagnitudes]

		# y = np.abs(yr[0:np.int(N/2)])
		# phases = [math.atan2(val.imag , val.real) for val in yr[0:np.int(N/2)]]
		
		# maxY = max(y)
		# index = np.where(y == [maxY])
		# maxMagnitude = max(magnitudes)
		# index = np.where(magnitudes == [maxMagnitude])
		# plt.plot(phases)
		print("max " , topMagnitudes, ", ", indexes)
		# print(magnitudes)
		return [int(topMagnitudes[0]),int(topMagnitudes[1]),int(topMagnitudes[2]),indexes[0], indexes[1], indexes[2]]

	def fftFull(self):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [float(x) for x in instance["pupilList"]]
			# print(temp)
			peakFFT = self.calculatePeakFrequency(temp, len(temp))
			print(instance["type"], ", " , peakFFT)
			print("---------------------------------------------------")
			# mean = sum(temp) / len(temp)
			# sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in temp]) / len(temp))
			# variance = math.sqrt((sum([(x * x) for x in temp]) - ((sum([x for x in temp]) * sum([x for x in temp]))/len(temp))) / (len(temp) - 1))
			peakFFT.extend([instance["baselineMean"]])
			feature.append(peakFFT)
			# break

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def movingWindowFFTFull(self, frameWidth):
		feature = []
		labels = []

		for instance in self.data:
			labels.append(instance["type"])
			temp = [float(x) for x in instance["pupilList"]]
			# print(temp)

			start = 0
			dataSeries = []
			while start < len(temp) - frameWidth:
				peakMagnitude = self.calculatePeakMagnitude(temp[start:start+frameWidth])
				dataSeries.append(peakMagnitude)
				start = start + 1
				pass

			# mean = sum(temp) / len(temp)
			# sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in temp]) / len(temp))
			# variance = math.sqrt((sum([(x * x) for x in temp]) - ((sum([x for x in temp]) * sum([x for x in temp]))/len(temp))) / (len(temp) - 1))
			dataSeries.append(instance["baselineMean"])
			feature.append(dataSeries)
			# break

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}
		pass