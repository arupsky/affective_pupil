from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt

class UsefulRegionChecker(object):
	"""docstring for UsefulRegionChecker"""
	def __init__(self):
		super(UsefulRegionChecker, self).__init__()
		self.inputFolder = "Formatter/Outputs/"
		self.frameWidth = 60
		self.interestZoneWidth = 180
		self.data = []
		self.evals = [[],[],[],[]]
		self.loadInputData()
		self.evaluateData()

	def loadInputData(self):
		dataFiles = [int(f.split('.')[0]) for f in listdir(self.inputFolder) if (isfile(join(self.inputFolder, f)) and ".json" in f)]
		if len(dataFiles) == 0:
			print("No input file found")
			return
		latestFile = str(max(dataFiles)) + ".json"
		
		with open(self.inputFolder + latestFile) as file:
			self.data = json.load(file)

	def evaluateData(self):
		# print(len(self.data))
		neg1 = [x["pupilList"] for x in self.data if x["type"] == 0]
		neg2 = [x["pupilList"] for x in self.data if x["type"] == 1]
		neu = [x["pupilList"] for x in self.data if x["type"] == 2]
		pos = [x["pupilList"] for x in self.data if x["type"] == 3]
		neu = neu[:len(neg1)]
		pos = neu[:len(neg1)]
		print(len(neg1), ", ", len(pos))
		self.calculate(neg1, self.evals[0], "Negative Stimuli - type 1")
		self.calculate(neg2, self.evals[1], "Negative Stimuli - type 2")
		self.calculate(neu, self.evals[2], "Neutral Stimuli")
		self.calculate(pos, self.evals[3], "Positive Stimuli")

		minimumError = 1000000
		minimumIndex = -1

		for i in range(len(self.evals[0])):
			mean = (self.evals[0][i] + self.evals[1][i] + self.evals[2][i] + self.evals[2][i]) / 4
			totalError = 0

			for j in range(4):
				totalError = totalError + abs(mean - self.evals[j][i])

			# print(totalError)
			if totalError < minimumError:
				minimumError = totalError
				minimumIndex = i

		print("most effective region ", minimumIndex, " - ", (minimumIndex + self.frameWidth))

		plt.show()

	def calculate(self, dataList, evaluationResult, title):
		
		for i in range(len(dataList)):
			firstValue = dataList[i][0]
			dataList[i] = [x - firstValue for x in dataList[i]]

		# start = 0
		meanStimuli = []
		totalErrors = []
		
		for i in range(self.interestZoneWidth):
			sizesThisFrame = [x[i] for x in dataList]
			meanValue = sum(sizesThisFrame) / len(sizesThisFrame)
			meanStimuli.append(meanValue)
			errorThisFrame = sum([abs(x - meanValue) for x in sizesThisFrame])
			totalErrors.append(errorThisFrame)
		
		
		start = 0

		while start < self.interestZoneWidth - self.frameWidth:
			evaluationResult.append(sum(totalErrors[start:start + self.frameWidth]))
			start = start + 1

		plt.figure()
		plt.subplot(311).set_title(title)
		for stimuli in dataList:
			plt.plot(stimuli)
		# plt.plot(dataList)
		plt.subplot(312).set_title("Mean")
		plt.plot(meanStimuli)

		plt.subplot(313).set_title("Total Error")
		plt.plot(totalErrors)
		
		
		# pass




		