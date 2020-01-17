import json
import matplotlib.pyplot as plt

def getMeanLine(data):
	count = len(data)
	lengths = [len(x) for x in data]
	length = min(lengths)
	mean_data = []
	for i in range(length):
		temp = []
		for sample in data:
			temp.append(sample[i])
		mean_data.append(sum(temp)/len(temp))

	return mean_data

def printForCsv(data, title):
	print("---------------", title, "----------------")
	for dt in data:
		print(dt)

folderIndividual2 = "data/individual2/"
folderIndividual3 = "data/individual3/"
folderIndividual4 = "data/individual4/"
folderGlobal2 = "data/global2/"
folderGlobal3 = "data/global3/"
folderGlobal4 = "data/global4/"

globalFolders = [folderGlobal2, folderGlobal3, folderGlobal4]
individualFolders = [folderIndividual2, folderIndividual3, folderIndividual4]
foldersInRoot = ["raw", "augmented", "augmented_train"]

trainFile = globalFolders[0] + "/raw/train.json"
testFile = globalFolders[0] + "/raw/test.json"

data = []

with open(trainFile) as file:
	data.extend(json.load(file))

with open(testFile) as file:
	data.extend(json.load(file))

classCount = 2

neutralImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 0]
affectiveImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 1]
neutralMean = getMeanLine(neutralImages)
affectiveMean = getMeanLine(affectiveImages)
xticks = [int(i * (1000/60)) for i in range(len(neutralMean))]
plt.plot(xticks, neutralMean, label='Neutral')
xticks = [int(i * (1000/60)) for i in range(len(affectiveMean))]
plt.plot(xticks, affectiveMean, label='Affective')
plt.xlabel('time (ms)')
plt.ylabel('pupil size (mm)')
plt.legend()
plt.show()



trainFile = globalFolders[1] + "/raw/train.json"
testFile = globalFolders[1] + "/raw/test.json"

data = []

with open(trainFile) as file:
	data.extend(json.load(file))

with open(testFile) as file:
	data.extend(json.load(file))

classCount = 3

negativeImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 0]
neutralImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 1]
positiveImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 2]

neutralMean = getMeanLine(neutralImages)
negativeMean = getMeanLine(negativeImages)
positiveMean = getMeanLine(positiveImages)



plt.figure()
xticks = [int(i * (1000/60)) for i in range(len(neutralMean))]
plt.plot(xticks, neutralMean, label='Neutral')
xticks = [int(i * (1000/60)) for i in range(len(positiveMean))]
plt.plot(xticks, positiveMean, label='Positive')
xticks = [int(i * (1000/60)) for i in range(len(negativeMean))]
plt.plot(xticks, negativeMean, label='Negative')
plt.xlabel('time (ms)')
plt.ylabel('pupil size (mm)')
plt.legend()
plt.show()


trainFile = globalFolders[2] + "/raw/train.json"
testFile = globalFolders[2] + "/raw/test.json"

data = []

with open(trainFile) as file:
	data.extend(json.load(file))

with open(testFile) as file:
	data.extend(json.load(file))

classCount = 3

negative1Images = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 0]
negative2Images = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 1]
neutralImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 2]
positiveImages = [dt["pupilListSmoothed"] for dt in data if dt["type"] == 3]

neutralMean = getMeanLine(neutralImages)
negative1Mean = getMeanLine(negative1Images)
negative2Mean = getMeanLine(negative2Images)
positiveMean = getMeanLine(positiveImages)

printForCsv(neutralMean, "Neutral")
printForCsv(negative1Mean, "Nega 1")
printForCsv(negative2Mean, "Nega 2")
printForCsv(positiveMean, "Positive")

plt.figure()
xticks = [int(i * (1000/60)) for i in range(len(neutralMean))]
plt.plot(xticks, neutralMean, label='Neutral')
xticks = [int(i * (1000/60)) for i in range(len(positiveMean))]
plt.plot(xticks, positiveMean, label='Positive')
xticks = [int(i * (1000/60)) for i in range(len(negative1Mean))]
plt.plot(xticks, negative1Mean, label='Human Violation')
xticks = [int(i * (1000/60)) for i in range(len(negative2Mean))]
plt.plot(xticks, negative2Mean, label='Animal Mistreatment')
plt.xlabel('time (ms)')
plt.ylabel('pupil size (mm)')
plt.legend()
plt.show()


