from os import listdir
from os import makedirs
from os.path import isfile, join

class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self):
		self.reportFolder = self.createNewFolder()

	def createNewFolder(self):

		folders = [int(f.split('.')[0]) for f in listdir("report") if not isfile(join("report", f))]
		if len(folders) == 0:
			outputFolderName = "0"
		else:
			outputFolderName = str(max(folders) + 1)
		print("Creating folder for output...")
		makedirs("report/" + outputFolderName)
		print("Folder creation successful. Output folder name : ", ("report/" + outputFolderName + "/"))
		return "report/" + outputFolderName + "/"
		