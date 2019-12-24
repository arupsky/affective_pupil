from SlicedDataFormatter import SlicedDataFormatter
from Helper import Helper
from FCNN_Re import FCNN_Re
import json
import tensorflow as tf

experiment = "MLP"
# folder = "augmented_5000"
# folder = "augmented"
folder = "raw"
# folder = "augmented_train_validation"
# folder = "rawtest2"
start = 0
window = 180

config = {}
config["epochs"] = 500
config["verbose"] = 1
config["batch_size"] = 64
config["layers"] = [15,6]

with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)


trainFeatures, trainLabels = Helper.getTrainableDataFCNN(trainSet, start, window)
fcnn = FCNN_Re(trainFeatures, trainLabels, config=config)

