from os import listdir
from os.path import isfile, join
import csv
import datetime
import json

class DataCollector(object):
	"""docstring for DataCollector"""
	def __init__(self, folderName):
		super(DataCollector, self).__init__()
		self.folderName = folderName
		
	def loadFolder(self):
		dataFiles = [f for f in listdir(self.folderName) if (isfile(join(self.folderName, f)) and ".csv" in f)]
		self.data = []
		for file in dataFiles:
			self.data.extend(self.extractData(file))
		return self.data


	def saveOutputFile(self):
		outputFolder = "DataCollector/Outputs/"
		outputFiles = [f for f in listdir(outputFolder) if (isfile(join(outputFolder, f)) and ".json" in f)]
		outputFileName = join(outputFolder, str(len(outputFiles)) + ".json")
		# print(outputFileName)
		with open(outputFileName, 'w') as outfile:
			json.dump(self.data, outfile)

	def extractData(self, fileName):
		data_list = []

		with open(self.folderName + fileName) as csvfile:
		    readCSV = csv.reader(csvfile, delimiter=',')
		    skip = True
		    for row in readCSV:
		        # print(row[10])
		        # print(row[12])
		        # print(row[14])
		        # print(row[15])
		        if skip or row[16] == "":
		        	skip = False
		        	# print("skipping")
		        	continue

		        pupilList = row[14][1:-1].split(', ')
		        pupilList = [float(x) for x in pupilList]
		        startSize = (pupilList[0] + pupilList[1] + pupilList[2] + pupilList[3])/4

		        data = {}
		        data["stimuliData"] = row[10]
		        data["type"] = int(row[16]) + 2 
		        data["baselineList"] = [float(x) for x in row[11][1:-1].split(', ')]
		        data["baselineMean"] = float(row[12])
		        data["pupilList"] = pupilList
		        # data["normalizedPupilList"] = [x / data["baselineMean"] for x in pupilList]
		        # data["zeroBasedPupilList"] = [x - data["normalizedPupilList"][0] for x in data["normalizedPupilList"]]
		        data["pupilMean"] = float(row[15])
		        data_list.append(data)

		        # if skip:
		        # 	skip = False
		        # 	break
		return data_list
		