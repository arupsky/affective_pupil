from os import listdir
from os.path import isfile, join
import json

class Formatter(object):

	def __init__(self):
		super(Formatter, self).__init__()
		self.inputFolder = "DataCollector/Outputs/"
		self.outputFolder = "Formatter/Outputs/"
		self.data = []
		

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
		return data


