from shutil import copyfile

def join(folderA, folderB):
	return folderA + "/" + folderB

def getFileName(exp, dataType, participant):
	return exp + "_" + dataType + "_" + str(participant) + ".png"

# copyfile(src, dst)

sourceRoot = "engine_results/8"
destinationRoot = "images"
rootFolders = ["class_2", "class_3", "class_4"]
dataTypes = ["raw", "augmented", "augmented_train"]
experiments = ["CNN", "MLP", "SlidingWindow"]

for folder in rootFolders:
	sourceFolder = join(join(sourceRoot, folder), "individual_classifier")
	destinationFolder = join(destinationRoot, folder)

	for participant in range(1,11):
		participantFolder = join(sourceFolder, str(participant))
		
		for dataType in dataTypes:
			dataFolder = join(participantFolder, dataType)

			for experiment in experiments:
				experimentFolder = join(dataFolder, experiment)
				sourceFile = join(experimentFolder, "confusionMatrixNormalized.png")
				destinationFile = join(destinationFolder, getFileName(experiment, dataType, participant))
				# print("source",sourceFile, "destination", destinationFile)
				copyfile(sourceFile, destinationFile)

