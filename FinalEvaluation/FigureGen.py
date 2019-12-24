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
xticks = [int(i * (1000/60)) for i in range(len(pplDt[:300]))]
ax.plot(xticks, pplDt)
plt.plot(neutralMean, label='Neutral')
plt.plot(affectiveMean, label='Affective')
plt.xlabel('time (ms)')
plt.ylabel('pupil size (mm)')
plt.legend()
plt.show()


