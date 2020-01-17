import json

def getResultObj(name, experiment):
	temp ={}
	temp["name"] = name
	temp["experiment"] = experiment
	temp["raw"] = 0
	temp["augmented"] = 0
	temp["augmented_train"] = 0

def getAccuracyResult(name):
	temp = {}
	temp["name"] = str(name)
	return temp

def getAccuracyString(accuracy):
	accu = float(accuracy) * 100
	return "{0:.2f}".format(accu) + "\\%"

resultFolder = "engine_results/8/"
classFolders = ["class_2/", "class_3/", "class_4/"]
classifierFolders = ["global_classifier/", "individual_classifier/"]
dataFolders = ["raw/", "augmented/", "augmented_train/"]
experimentFolders = ["CNN/", "MLP/", "SlidingWindow/"]

output = {}

for className in classFolders:
	classFolder = resultFolder + className

	classifierFolder = classFolder + "global_classifier/"

	cnnAccuracies = []
	mlpAccuracies = []
	swAccuracies = []
	titles = []
	print(className)

	for dataType in dataFolders:
		titles.append(dataType)
		dataFolder = classifierFolder + dataType
		fileName = dataFolder + "report.json"
		# print("Global " + dataType)
		with open(fileName) as file:
			data = json.load(file)
			
			cnnResults = [dt for dt in data if dt["experiment"] == "CNN"]
			cnnAccuracies.append(cnnResults[0]["accuracy"])

			mlpResults = [dt for dt in data if dt["experiment"] == "MLP"]
			mlpAccuracies.append(mlpResults[0]["accuracy"])

			swResults = [dt for dt in data if dt["experiment"] == "SlidingWindow"]
			swAccuracies.append(swResults[0]["accuracy"])


	classifierFolderRoot = classFolder + "individual_classifier/"
	cnnAccuraciesIndi = []
	mlpAccuraciesIndi = []
	swAccuraciesIndi = []

	for i in range(1,11):
		classifierFolder = classifierFolderRoot + str(i) + "/"

		accuCnn = getAccuracyResult(i)
		accuMlp = getAccuracyResult(i)
		accuSw = getAccuracyResult(i)
		for dataType in dataFolders:
			dataFolder = classifierFolder + dataType
			fileName = dataFolder + "report.json"
			# print("Global " + dataType)
			with open(fileName) as file:
				data = json.load(file)
				
				cnnResults = [dt for dt in data if dt["experiment"] == "CNN"]
				# cnnAccuracies.append(cnnResults[0]["accuracy"])
				accuCnn[dataType[:-1]] = cnnResults[0]["accuracy"]

				mlpResults = [dt for dt in data if dt["experiment"] == "MLP"]
				# mlpAccuracies.append(mlpResults[0]["accuracy"])
				accuMlp[dataType[:-1]] = mlpResults[0]["accuracy"]

				swResults = [dt for dt in data if dt["experiment"] == "SlidingWindow"]
				# swAccuracies.append(swResults[0]["accuracy"])
				accuSw[dataType[:-1]] = swResults[0]["accuracy"]
		cnnAccuraciesIndi.append(accuCnn)
		mlpAccuraciesIndi.append(accuMlp)
		swAccuraciesIndi.append(accuSw)


	print("", titles)
	print("Global & ",getAccuracyString(cnnAccuracies[0]),"&",getAccuracyString(cnnAccuracies[1]),"&",getAccuracyString(cnnAccuracies[2]),"\\\\")
	for accu in cnnAccuraciesIndi:
		print("Participant",accu["name"],"&",getAccuracyString(accu['raw']),"&",getAccuracyString(accu["augmented"]),"&",getAccuracyString(accu["augmented_train"]),"\\\\")
	
	print("Global & ",getAccuracyString(mlpAccuracies[0]),"&",getAccuracyString(mlpAccuracies[1]),"&",getAccuracyString(mlpAccuracies[2]),"\\\\")
	for accu in mlpAccuraciesIndi:
		print("Participant",accu["name"],"&",getAccuracyString(accu['raw']),"&",getAccuracyString(accu["augmented"]),"&",getAccuracyString(accu["augmented_train"]),"\\\\")
	
	print("Global & ",getAccuracyString(swAccuracies[0]),"&",getAccuracyString(swAccuracies[1]),"&",getAccuracyString(swAccuracies[2]),"\\\\")
	for accu in swAccuraciesIndi:
		print("Participant",accu["name"],"&",getAccuracyString(accu['raw']),"&",getAccuracyString(accu["augmented"]),"&",getAccuracyString(accu["augmented_train"]),"\\\\")
	print("-----------------")

