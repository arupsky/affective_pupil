from os import listdir
from os.path import isfile, join
import json
import numpy as np
from Helper import Helper

class Formatter(object):

	def __init__(self, globalConfig):
		super(Formatter, self).__init__()
		self.inputFolder = "DataCollector/Outputs/"
		self.outputFolder = "Formatter/Outputs/"
		self.data = []
		self.interestingZone = 300
		self.baselineEffectZone = 60
		self.globalConfig = globalConfig
		

	def formatLatest(self):
		self.loadInputData()
		processedData = self.process(self.data)
		self.saveOutputFile(processedData)
		

	def loadInputData(self):
		dataFiles = [int(f.split('.')[0]) for f in listdir(self.inputFolder) if (isfile(join(self.inputFolder, f)) and ".json" in f)]
		if len(dataFiles) == 0:
			print("No input file found")
			return
		latestFile = str(max(dataFiles)) + ".json"
		
		with open(self.inputFolder + latestFile) as file:
			self.data = json.load(file)

	def saveOutputFile(self, data):
		outputFiles = [int(f.split('.')[0]) for f in listdir(self.outputFolder) if (isfile(join(self.outputFolder, f)) and ".json" in f)]
		latestOutputFile = str(len(outputFiles)) + ".json"
		print(latestOutputFile)

		with open(self.outputFolder + latestOutputFile, 'w') as file:
			json.dump(data, file)

	# data from useful region only are extracted here
	def process(self, data):

		processed = []

		for dt in data:
			temp = {}
			temp["type"] = dt["type"]
			temp["baselineList"] = dt["baselineList"]
			if self.globalConfig["minimum_as_baseline"]:
				minimumDilation = min(dt["pupilListSmoothed"][:60])
				minimumIndex = np.where(np.array(dt["pupilListSmoothed"][:60]) == minimumDilation)[0][0]
				temp["baselineMean"] = minimumDilation
				temp["pupilListSmoothed"] = [x - minimumDilation for x in dt["pupilListSmoothed"][minimumIndex + 1:self.interestingZone + minimumIndex + 1]]
			elif self.globalConfig["fixation_as_baseline"]:
				baselineArea = dt["baselineList"][-60:]
				temp["baselineMean"] = sum(baselineArea)/len(baselineArea)
				temp["pupilListSmoothed"] = [x - temp["baselineMean"] for x in dt["pupilListSmoothed"][:self.interestingZone]]
			elif self.globalConfig["fixation_minimum_interpolate"]:
				centralBaseline = Helper.getCentralBaselineMean(dt["baselineList"])
				minimumVal, minimumIndex = Helper.getFirstMinimumValueAndIndex(dt["pupilListSmoothed"])
				diff = (minimumVal - centralBaseline) / (minimumIndex + 1)
				newPupilList = [centralBaseline] * 5
				for i in range(minimumIndex):
					interpolatedVal = centralBaseline + diff * i
					newPupilList.append(interpolatedVal)
				newPupilList.extend(dt["pupilListSmoothed"][minimumIndex:])
				# newPupilList = Helper.smooth(newPupilList, 3)
				temp["pupilListSmoothed"] = [x - centralBaseline for x in newPupilList][:self.interestingZone]
				temp["baselineMean"] = centralBaseline

			else:
				temp["baselineMean"] = sum(temp["baselineList"][-self.baselineEffectZone:])/self.baselineEffectZone
				temp["pupilListSmoothed"] = [x - temp["baselineMean"] for x in dt["pupilListSmoothed"][:self.interestingZone]]

			# temp["pupilSkewness"] = Helper.getWholeSkewness(temp["pupilListSmoothed"], 5)
			# temp["pupilKurtosis"] = Helper.getWholeKurtosis(temp["pupilListSmoothed"], 5)
			temp["pupilMean"] = sum(temp["pupilListSmoothed"])/self.interestingZone
			temp["imageName"] = dt["imageName"]
			processed.append(temp)


		return processed

	def processTsvData(self, data):
		pass


