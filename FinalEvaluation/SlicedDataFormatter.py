from os import listdir
from os import makedirs
from os.path import isfile, join
import json

class SlicedDataFormatter:
	"""docstring for SlicedDataFormatter"""
	def __init__(self, folderName):
		self.folderName = folderName
		self.process()

	def process(self):
		print(listdir(self.folderName))
		

		for file in listdir(self.folderName):
			with open(self.folderName + "/" + file) as fl:
				jsonData = json.load(fl)
				print("length", len(jsonData))
				print(jsonData[0]["pupilListSmoothed"][:5])
		