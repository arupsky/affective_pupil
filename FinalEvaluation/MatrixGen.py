import numpy as np
from Helper import Helper
import matplotlib.pyplot as plt

def join(folderA, folderB):
	return folderA + "/" + folderB

def getFileName(exp, dataType, participant):
	return exp + "_" + dataType + "_" + str(participant) + ".png"

def isValid(data, experiment, className, dataType):
	return data["experiment"] == experiment and data["class"] == className and data["dataType"] == dataType

def isGlobalValid(data, experiment, className):
	return data["experiment"] == experiment and data["class"] == className

sourceRoot = "engine_results/12"
rootFolders = ["class_2", "class_3", "class_4"]
dataTypes = ["raw", "augmented", "augmented_train"]
experiments = ["CNN", "MLP", "SlidingWindow"]

matrices = []
globalMatrices = []

for folder in rootFolders:
	sourceFolder = join(join(sourceRoot, folder), "individual_classifier")

	for participant in range(1,11):
		participantFolder = join(sourceFolder, str(participant))
		
		for dataType in dataTypes:
			dataFolder = join(participantFolder, dataType)

			for experiment in experiments:
				experimentFolder = join(dataFolder, experiment)
				sourceFile = join(experimentFolder, "matrix.npy")
				temp = {}
				temp["class"] = folder
				temp["dataType"] = dataType
				temp["experiment"] = experiment
				temp["matrix"] = np.load(sourceFile)
				temp["participant"] = participant
				matrices.append(temp)

for folder in rootFolders:
	sourceFolder = join(join(sourceRoot, folder), "global_classifier")

	for dataType in dataTypes:
		dataFolder = join(sourceFolder, dataType)

		for experiment in experiments:
			experimentFolder = join(dataFolder, experiment)
			sourceFile = join(experimentFolder, "matrix.npy")
			temp = {}
			temp["class"] = folder
			temp["dataType"] = dataType
			temp["experiment"] = experiment
			temp["matrix"] = np.load(sourceFile)
			globalMatrices.append(temp)


# cnn_class2_raw = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_2", "raw")]
# cnn_class2_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_2", "augmented")]
# cnn_class2_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_2", "augmented_train")]

# cnn_class3_raw = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_3", "raw")]
# cnn_class3_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_3", "augmented")]
# cnn_class3_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_3", "augmented_train")]

# cnn_class4_raw = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_4", "raw")]
# cnn_class4_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_4", "augmented")]
# cnn_class4_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "CNN", "class_4", "augmented_train")]


# Helper.plotConfusionMatrices(cnn_class2_raw, ["Neutral", "Affective"], fileName = 'cnn_class2_raw', title="CNN trained with Raw Data")
# Helper.plotConfusionMatrices(cnn_class2_augmented, ["Neutral", "Affective"], fileName = 'cnn_class2_augmented', title="CNN trained with Augmented Data")
# Helper.plotConfusionMatrices(cnn_class2_augmented_train, ["Neutral", "Affective"], fileName = 'cnn_class2_augmented_train', title="CNN trained with Augmented Train Data")

# Helper.plotConfusionMatrices(cnn_class3_raw, ["Negative","Neutral", "Positive"], fileName = 'cnn_class3_raw', title="CNN trained with Raw Data")
# Helper.plotConfusionMatrices(cnn_class3_augmented, ["Negative","Neutral", "Positive"], fileName = 'cnn_class3_augmented', title="CNN trained with Augmented Data")
# Helper.plotConfusionMatrices(cnn_class3_augmented_train, ["Negative","Neutral", "Positive"], fileName = 'cnn_class3_augmented_train', title="CNN trained with Augmented Train Data")

# Helper.plotConfusionMatrices(cnn_class4_raw, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'cnn_class4_raw', title="CNN trained with Raw Data")
# Helper.plotConfusionMatrices(cnn_class4_augmented, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'cnn_class4_augmented', title="CNN trained with Augmented Data")
# Helper.plotConfusionMatrices(cnn_class4_augmented_train, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'cnn_class4_augmented_train', title="CNN trained with Augmented Train Data")


# mlp_class2_raw = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_2", "raw")]
# mlp_class2_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_2", "augmented")]
# mlp_class2_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_2", "augmented_train")]

# mlp_class3_raw = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_3", "raw")]
# mlp_class3_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_3", "augmented")]
# mlp_class3_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_3", "augmented_train")]

# mlp_class4_raw = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_4", "raw")]
# mlp_class4_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_4", "augmented")]
# mlp_class4_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "MLP", "class_4", "augmented_train")]


# Helper.plotConfusionMatrices(mlp_class2_raw, ["Neutral", "Affective"], fileName = 'mlp_class2_raw', title="MLP trained with Raw Data")
# Helper.plotConfusionMatrices(mlp_class2_augmented, ["Neutral", "Affective"], fileName = 'mlp_class2_augmented', title="MLP trained with Augmented Data")
# Helper.plotConfusionMatrices(mlp_class2_augmented_train, ["Neutral", "Affective"], fileName = 'mlp_class2_augmented_train', title="MLP trained with Augmented Train Data")

# Helper.plotConfusionMatrices(mlp_class3_raw, ["Negative","Neutral", "Positive"], fileName = 'mlp_class3_raw', title="MLP trained with Raw Data")
# Helper.plotConfusionMatrices(mlp_class3_augmented, ["Negative","Neutral", "Positive"], fileName = 'mlp_class3_augmented', title="MLP trained with Augmented Data")
# Helper.plotConfusionMatrices(mlp_class3_augmented_train, ["Negative","Neutral", "Positive"], fileName = 'mlp_class3_augmented_train', title="MLP trained with Augmented Train Data")

# Helper.plotConfusionMatrices(mlp_class4_raw, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'mlp_class4_raw', title="MLP trained with Raw Data")
# Helper.plotConfusionMatrices(mlp_class4_augmented, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'mlp_class4_augmented', title="MLP trained with Augmented Data")
# Helper.plotConfusionMatrices(mlp_class4_augmented_train, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'mlp_class4_augmented_train', title="MLP trained with Augmented Train Data")

# sw_class2_raw = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_2", "raw")]
# sw_class2_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_2", "augmented")]
# sw_class2_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_2", "augmented_train")]

# sw_class3_raw = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_3", "raw")]
# sw_class3_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_3", "augmented")]
# sw_class3_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_3", "augmented_train")]

# sw_class4_raw = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_4", "raw")]
# sw_class4_augmented = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_4", "augmented")]
# sw_class4_augmented_train = [temp["matrix"] for temp in matrices if isValid(temp, "SlidingWindow", "class_4", "augmented_train")]


# Helper.plotConfusionMatrices(sw_class2_raw, ["Neutral", "Affective"], fileName = 'sw_class2_raw', title="MLP (Sliding Window) trained with Raw Data")
# Helper.plotConfusionMatrices(sw_class2_augmented, ["Neutral", "Affective"], fileName = 'sw_class2_augmented', title="MLP (Sliding Window) trained with Augmented Data")
# Helper.plotConfusionMatrices(sw_class2_augmented_train, ["Neutral", "Affective"], fileName = 'sw_class2_augmented_train', title="MLP (Sliding Window) trained with Augmented Train Data")

# Helper.plotConfusionMatrices(sw_class3_raw, ["Negative","Neutral", "Positive"], fileName = 'sw_class3_raw', title="MLP (Sliding Window) trained with Raw Data")
# Helper.plotConfusionMatrices(sw_class3_augmented, ["Negative","Neutral", "Positive"], fileName = 'sw_class3_augmented', title="MLP (Sliding Window) trained with Augmented Data")
# Helper.plotConfusionMatrices(sw_class3_augmented_train, ["Negative","Neutral", "Positive"], fileName = 'sw_class3_augmented_train', title="MLP (Sliding Window) trained with Augmented Train Data")

# Helper.plotConfusionMatrices(sw_class4_raw, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'sw_class4_raw', title="MLP (Sliding Window) trained with Raw Data")
# Helper.plotConfusionMatrices(sw_class4_augmented, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'sw_class4_augmented', title="MLP (Sliding Window) trained with Augmented Data")
# Helper.plotConfusionMatrices(sw_class4_augmented_train, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'sw_class4_augmented_train', title="MLP (Sliding Window) trained with Augmented Train Data")


global_cnn_class2 = [temp for temp in globalMatrices if isGlobalValid(temp, "CNN", "class_2")]
global_cnn_class3 = [temp for temp in globalMatrices if isGlobalValid(temp, "CNN", "class_3")]
global_cnn_class4 = [temp for temp in globalMatrices if isGlobalValid(temp, "CNN", "class_4")]

Helper.plotGlobalConfusionMatrices(global_cnn_class2, ["Neutral", "Affective"], fileName = 'global_cnn_class2')
Helper.plotGlobalConfusionMatrices(global_cnn_class3, ["Negative","Neutral", "Positive"], fileName = 'global_cnn_class3')
Helper.plotGlobalConfusionMatrices(global_cnn_class4, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'global_cnn_class4')

global_mlp_class2 = [temp for temp in globalMatrices if isGlobalValid(temp, "MLP", "class_2")]
global_mlp_class3 = [temp for temp in globalMatrices if isGlobalValid(temp, "MLP", "class_3")]
global_mlp_class4 = [temp for temp in globalMatrices if isGlobalValid(temp, "MLP", "class_4")]

Helper.plotGlobalConfusionMatrices(global_mlp_class2, ["Neutral", "Affective"], fileName = 'global_mlp_class2')
Helper.plotGlobalConfusionMatrices(global_mlp_class3, ["Negative","Neutral", "Positive"], fileName = 'global_mlp_class3')
Helper.plotGlobalConfusionMatrices(global_mlp_class4, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'global_mlp_class4')

global_sw_class2 = [temp for temp in globalMatrices if isGlobalValid(temp, "SlidingWindow", "class_2")]
global_sw_class3 = [temp for temp in globalMatrices if isGlobalValid(temp, "SlidingWindow", "class_3")]
global_sw_class4 = [temp for temp in globalMatrices if isGlobalValid(temp, "SlidingWindow", "class_4")]

Helper.plotGlobalConfusionMatrices(global_sw_class2, ["Neutral", "Affective"], fileName = 'global_sw_class2')
Helper.plotGlobalConfusionMatrices(global_sw_class3, ["Negative","Neutral", "Positive"], fileName = 'global_sw_class3')
Helper.plotGlobalConfusionMatrices(global_sw_class4, ["Negative 1","Negative 2","Neutral", "Positive"], fileName = 'global_sw_class4')

plt.show()
