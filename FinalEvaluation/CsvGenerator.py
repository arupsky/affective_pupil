import json

folder = "data/global3/raw"

trainFile = folder + "/train.json"
testFile = folder + "/test.json"

allData = []

with open(trainFile) as file:
	allData.extend(json.load(file))

with open(testFile) as file:
	allData.extend(json.load(file))

print(len([x for x in allData if x["type"] == 1]))


for i in range(1,11):
	with open("data/source/"+str(i)+".json") as file:
		rawData = json.load(file)
		print(len(rawData))
		csv = "Frame\\type"
		comma = True
		frameNumber = 0

		for data in rawData:
			if comma:
				csv = csv + ","
			csv = csv + str(data["type"])
			comma = True
		csv = csv + '\n' + str(frameNumber)
		frameNumber = frameNumber + 1
		for frame in range(300):
			comma = True

			for data in rawData:
				if frame >= len(data["pupilListSmoothed"]):
					continue
				if comma:
					csv = csv + ","
				csv = csv + str(data["pupilListSmoothed"][frame])
				comma = True
			csv = csv + '\n' + str(frameNumber)
			frameNumber = frameNumber + 1

		with open("data/csvs/" + str(i)+".csv", "w") as output:
			output.write(csv)
