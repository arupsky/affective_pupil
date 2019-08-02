import csv
import matplotlib.pyplot as plt


def extractData(fileName):
	data_list = []

	with open("../../data/" + fileName) as csvfile:
	    readCSV = csv.reader(csvfile, delimiter=',')
	    skip = True
	    for row in readCSV:
	        # print(row[10])
	        # print(row[12])
	        # print(row[14])
	        # print(row[15])
	        if skip or row[16] == "":
	        	skip = False
	        	print("skipping")
	        	continue

	        pupilList = row[14][1:-1].split(', ')
	        pupilList = [float(x) for x in pupilList]
	        startSize = (pupilList[0] + pupilList[1] + pupilList[2] + pupilList[3])/4

	        data = {}
	        data["stimuliData"] = row[10]
	        data["type"] = int(row[16]) + 2 
	        data["baselineMean"] = float(row[12])
	        data["pupilList"] = pupilList
	        data["normalizedPupilList"] = [x - data["baselineMean"] for x in pupilList]
	        data["zeroBasedPupilList"] = [x - data["normalizedPupilList"][0] for x in data["normalizedPupilList"]]
	        data["pupilMean"] = float(row[15])
	        data_list.append(data)

	        # if skip:
	        # 	skip = False
	        # 	break
	return data_list

def plotFirst100Points(pupilData, key, name):
	f = plt.figure(name)
	for dt in pupilData:
		plt.plot(dt[key][:150])
	

data = extractData("1219_untitled_2019_Jul_01_1126.csv")
# print(data)
# plt.plot(data[0]["normalizedPupilList"])
# plt.show()

negativeData1 = [x for x in data if x["type"]==0]
negativeData2 = [x for x in data if x["type"]==1]
neutralData = [x for x in data if x["type"]==2]
positiveData = [x for x in data if x["type"]==3]
# print([x["type"] for x in negativeData1])
# for dt in negativeData1:
# 	plt.plot(dt["zeroBasedPupilList"])

# plt.show();

plotFirst100Points(negativeData1, "zeroBasedPupilList", "negativeData1")
plotFirst100Points(negativeData2, "zeroBasedPupilList", "negativeData2")
plotFirst100Points(neutralData, "zeroBasedPupilList", "neutralData")
plotFirst100Points(positiveData, "zeroBasedPupilList", "positiveData")


plt.show();