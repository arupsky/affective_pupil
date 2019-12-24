import json
import random
from os import makedirs

input_file_name = "data/augmented_train_raw.json"
output_folder = "data/splitted/rawtest2"

with open(input_file_name) as inpuFile:
	data = json.load(inpuFile)

	totalCount = len(data)
	avgCountPerClass = int(totalCount/3)
	trainCountPerClass = int(avgCountPerClass * .67)
	# validationCountPerClass = int(avgCountPerClass * .33)

	trainSet = []
	validationSet = []
	testSet = []

	negativeFormatted = [x for x in data if x["type"] == 0]
	neutralFormatted = [x for x in data if x["type"] == 1]
	positiveFormatted = [x for x in data if x["type"] == 2]

	negativeTrain = random.sample(negativeFormatted, trainCountPerClass)
	neutralTrain = random.sample(neutralFormatted, trainCountPerClass)
	positiveTrain = random.sample(positiveFormatted, trainCountPerClass)

	negativeRemaining = [x for x in negativeFormatted if x not in negativeTrain]
	neutralRemaining = [x for x in neutralFormatted if x not in neutralTrain]
	positiveRemaining = [x for x in positiveFormatted if x not in positiveTrain]

	# negativeValidation = random.sample(negativeRemaining, validationCountPerClass)
	# neutralValidation = random.sample(neutralRemaining, validationCountPerClass)
	# positiveValidation = random.sample(positiveRemaining, validationCountPerClass)

	# testSet.extend([x for x in negativeRemaining if x not in negativeValidation])
	# testSet.extend([x for x in neutralRemaining if x not in neutralValidation])
	# testSet.extend([x for x in positiveRemaining if x not in positiveValidation])

	validationSet.extend(negativeRemaining)
	validationSet.extend(neutralRemaining)
	validationSet.extend(positiveRemaining)

	trainSet.extend(negativeTrain)
	trainSet.extend(neutralTrain)
	trainSet.extend(positiveTrain)

	# print("total", totalCount)
	# print("avgCountPerClass", avgCountPerClass)
	# print("trainCountPerClass", trainCountPerClass)
	# print("validationCountPerClass",validationCountPerClass)
	# print("--")
	# print("negativeFormatted", len(negativeFormatted))
	# print("neutralFormatted", len(neutralFormatted))
	# print("positiveFormatted", len(positiveFormatted))
	# print("--")
	# print("negativeTrain", len(negativeTrain))
	# print("neutralTrain", len(neutralTrain))
	# print("positiveTrain", len(positiveTrain))
	# print("--")
	# print("negativeRemaining", len(negativeRemaining))
	# print("neutralRemaining", len(neutralRemaining))
	# print("positiveRemaining", len(positiveRemaining))
	# print("--")
	# print("--")
	# print("negativeValidation", len(negativeValidation))
	# print("neutralValidation", len(neutralValidation))
	# print("positiveValidation", len(positiveValidation))
	# print("Test set info")
	# print("negative count ",sum([1 for x in testSet if x["type"] == 0]))
	# print("negative count ",sum([1 for x in testSet if x["type"] == 1]))
	# print("negative count ",sum([1 for x in testSet if x["type"] == 2]))

	makedirs(output_folder)
	with open(output_folder + "/train.json", 'w') as outFile:
		json.dump(trainSet, outFile)

	with open(output_folder + "/validation.json", 'w') as outFile:
		json.dump(validationSet, outFile)

	# with open(output_folder + "/test.json", 'w') as outFile:
	# 	json.dump(testSet, outFile)



