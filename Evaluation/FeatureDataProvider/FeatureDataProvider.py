from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import math  

class FeatureDataProvider(object):
	"""docstring for FeatureDataProvider"""
	def __init__(self, fileName=""):
		super(FeatureDataProvider, self).__init__()
		self.fileName = fileName

		self.inputFolder = "Formatter/Outputs/"
		self.outputFolderRoot = "FeatureDataProvider/Outputs/"
		self.data = []
		self.loadInputData()
		self.process()
		plt.show()


	def loadInputData(self):
		dataFiles = [int(f.split('.')[0]) for f in listdir(self.inputFolder) if (isfile(join(self.inputFolder, f)) and ".json" in f)]
		if len(dataFiles) == 0:
			print("No input file found")
			return
		latestFile = str(max(dataFiles)) + ".json"
		
		with open(self.inputFolder + latestFile) as file:
			self.data = json.load(file)

		print(len(self.data))


	def saveOutputFile(self, saveData, fileName):
		print(fileName, ", ", saveData)
		plt.figure(fileName)
		for dt in saveData["features"]:
			plt.plot(dt)

	def getMeanSdVariance(self, dataSet):

		mean = sum(dataSet) / len(dataSet)
		sd = math.sqrt(sum([((x - mean) * (x - mean)) for x in dataSet]) / len(dataSet))
		variance = math.sqrt((sum([(x * x) for x in dataSet]) - ((sum([x for x in dataSet]) * sum([x for x in dataSet]))/len(dataSet))) / (len(dataSet) - 1))

		return [mean,sd,variance]

	def process(self):
		print(self.getMeanSdVariance([1,2,0,3]))
		# create new folder for output files
		newFolderName = "temp"
		# self.saveOutputFile(self.rawDataAtSecond(0), "raw_s_1") #60
		# self.saveOutputFile(self.rawDataAtSecond(1), "raw_s_2") #60
		# self.saveOutputFile(self.rawDataAtSecond(2), "raw_s_3") #60

		# self.saveOutputFile(self.normalizedDataAtSecond(0), "norm_s_1") #60
		# self.saveOutputFile(self.normalizedDataAtSecond(1), "norm_s_2") #60
		# self.saveOutputFile(self.normalizedDataAtSecond(2), "norm_s_3") #60
		# self.saveOutputFile(self.normalizedFull(), "norm")

		# self.saveOutputFile(self.commonFeatureAtSecond(0), "common_s_1") # mean, sd, variance
		# self.saveOutputFile(self.commonFeatureAtSecond(1), "common_s_2") # mean, sd, variance
		# self.saveOutputFile(self.commonFeatureAtSecond(2), "common_s_3") # mean, sd, variance
		# self.saveOutputFile(self.commonFeaturesFull(), "common") # mean, sd, variance

		# self.saveOutputFile(self.fftAtSecond(0), "fft_s_1")
		# self.saveOutputFile(self.fftAtSecond(1), "fft_s_2")
		# self.saveOutputFile(self.fftAtSecond(2), "fft_s_3")
		# self.saveOutputFile(self.fftFull(), "fft")

		# self.saveOutputFile(self.movingWindowFFTAtSecond(0, 8), "window_fft_s_1_8")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(0, 16), "window_fft_s_1_16")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(0, 32), "window_fft_s_1_32")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(1, 8), "window_fft_s_2_8")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(1, 16), "window_fft_s_2_16")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(1, 32), "window_fft_s_2_32")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(2, 8), "window_fft_s_3_8")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(2, 16), "window_fft_s_3_16")
		# self.saveOutputFile(self.movingWindowFFTAtSecond(2, 32), "window_fft_s_3_32")


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

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def fftAtSecond(self, second):
		pass

	def movingWindowFFTAtSecond(self, second, frameWidth):
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

		return {
			"features" 	: 	feature,
			"labels"	:	labels
		}

	def fftFull(self):
		pass

	def movingWindowFFTFull(self, frameWidth):
		pass