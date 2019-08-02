from os import listdir
from os.path import isfile, join
import csv

class CSVParser:
	"""docstring for CSVParser"""
	def __init__(self):
		folderName = "../../data/"
		dataFiles = [f for f in listdir(folderName) if (isfile(join(folderName, f)) and ".csv" in f)]
		self.data = []
		print(dataFiles)
		for file in dataFiles:
			self.data.extend(self.extractData(file))
		self.negativeData1 = [x for x in self.data if x["type"]==0]
		self.negativeData2 = [x for x in self.data if x["type"]==1]
		self.neutralData = [x for x in self.data if x["type"]==2]
		self.positiveData = [x for x in self.data if x["type"]==3]


	def getNegativeData1(self):
		return self.negativeData1

	def getNegativeData2(self):
		return self.negativeData2

	def getNegativeData(self):
		allNegativeData = []
		allNegativeData.extend(self.negativeData1)
		allNegativeData.extend(self.negativeData2)
		return allNegativeData

	def getPositiveData(self):
		return self.positiveData

	def getNeutralData(self):
		return self.neutralData

	def getDataForKey(self, dataList, key):
		result = [x[key] for x in dataList]
		return result

	def extractData(self, fileName):
		data_list = []

		with open("../../data/" + fileName) as csvfile:
		    readCSV = csv.reader(csvfile, delimiter=',')
		    skip = True
		    for row in readCSV:
		        # print(row[10])
		        # print(row[12])
		        # print(row[14])
		        # print(row[15])
		        if skip or row[16] == "":
		        	skip = False
		        	print("skipping")
		        	continue

		        pupilList = row[14][1:-1].split(', ')
		        pupilList = [float(x) for x in pupilList]
		        startSize = (pupilList[0] + pupilList[1] + pupilList[2] + pupilList[3])/4

		        data = {}
		        data["stimuliData"] = row[10]
		        data["type"] = int(row[16]) + 2 
		        data["baselineMean"] = float(row[12])
		        data["pupilList"] = pupilList
		        data["normalizedPupilList"] = [x / data["baselineMean"] for x in pupilList]
		        data["zeroBasedPupilList"] = [x - data["normalizedPupilList"][0] for x in data["normalizedPupilList"]]
		        data["pupilMean"] = float(row[15])
		        data_list.append(data)

		        # if skip:
		        # 	skip = False
		        # 	break
		return data_list
		