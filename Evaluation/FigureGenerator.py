from Formatter.Formatter import Formatter
from FeatureDataProvider.FeatureDataProvider import FeatureDataProvider
from DataCollector.DataCollector import DataCollector
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import math
import numpy as np

# formatter = Formatter()
# formatter.loadInputData()
# processedData = formatter.process(formatter.data)
# print(processedData)
# for data in processedData:
# 	plt.plot(data["pupilList"])

# featureProvider = FeatureDataProvider()
# processedData = featureProvider.commonFeatureAtSecond(1)
# # print(processedData)

# sds = [dt[1] for dt in processedData["features"]]
# print(sorted(sds))
# print("Max sd ", max(sds))

# # for data in processedData["features"]:
# # 	plt.plot(data)

# plt.show()


dataCollector = DataCollector("../../data/")
formatter = Formatter()
featureProvider = FeatureDataProvider()

data = dataCollector.extractData("1219_untitled_2019_Jul_01_1126.csv")
formattedData = formatter.process(data)

# formatter.loadInputData()
# formattedData = formatter.process(formatter.data)

outlierFreeData = [dt for dt in formattedData if featureProvider.getMeanSdVariance(dt["pupilList"])[1] < .5]
positiveData = [dt for dt in outlierFreeData if dt["type"] == 3]
negativeData = [dt for dt in outlierFreeData if dt["type"] < 2]
neutralData = [dt for dt in outlierFreeData if dt["type"] == 2]

positiveDataNormalized = [[y - sum(dt["baselineList"][-30:])/30 for y in dt["pupilList"]] for dt in positiveData]
negativeDataNormalized = [[y - sum(dt["baselineList"][-30:])/30 for y in dt["pupilList"]] for dt in negativeData]
neutralDataNormalized = [[y - sum(dt["baselineList"][-30:])/30 for y in dt["pupilList"]] for dt in neutralData]



pupilDataCount = len(outlierFreeData[0]["pupilList"])

meanPositive = []
meanNegative = []
meanNeutral = []

meanPositiveSmooth = []
meanNegativeSmooth = []
meanNeutralSmooth = []

smoothNeutral = [featureProvider.smooth(dt["pupilList"], 7) for dt in neutralData]
smoothPositive = [featureProvider.smooth(dt["pupilList"], 7) for dt in positiveData]
smoothNegative = [featureProvider.smooth(dt["pupilList"], 7) for dt in negativeData]


# mean normalized
for i in range(pupilDataCount):
	meanNegative.append(sum([dt["pupilList"][i] - sum(dt["baselineList"][-30:])/30 for dt in negativeData])/len(negativeData))
	meanPositive.append(sum([dt["pupilList"][i] - sum(dt["baselineList"][-30:])/30 for dt in positiveData])/len(positiveData))
	meanNeutral.append(sum([dt["pupilList"][i] - sum(dt["baselineList"][-30:])/30 for dt in neutralData])/len(neutralData))

# # mean normalized 2
# for i in range(pupilDataCount):
# 	meanNegative.append(sum([dt["pupilList"][i] for dt in negativeDataNormalized])/len(negativeDataNormalized))
# 	meanPositive.append(sum([dt["pupilList"][i] for dt in positiveDataNormalized])/len(positiveDataNormalized))
# 	meanNeutral.append(sum([dt["pupilList"][i] for dt in neutralDataNormalized])/len(neutralDataNormalized))


# mean normalized
for i in range(pupilDataCount):
	meanNegativeSmooth.append(sum([dt[i] - sum(dt[-30:])/30 for dt in smoothNegative])/len(smoothNegative))
	meanPositiveSmooth.append(sum([dt[i] - sum(dt[-30:])/30 for dt in smoothPositive])/len(smoothPositive))
	meanNeutralSmooth.append(sum([dt[i] - sum(dt[-30:])/30 for dt in smoothNeutral])/len(smoothNeutral))



plt.figure("Baseline corrected mean (individual)")
plt.plot(featureProvider.smooth(meanNeutral, 7), 'b', label="Neutral")
plt.plot(featureProvider.smooth(meanPositive, 7), 'g', label="Positive")
plt.plot(featureProvider.smooth(meanNegative, 7), 'r', label="Negative")
plt.xlabel("time (1/60 s)")
plt.ylabel("mean pupil diameter change (mm)")
plt.legend()

plt.figure("Baseline corrected mean (first 2 seconds) (individual)")
plt.plot(featureProvider.smooth(meanNeutral, 7)[:120],'b', label="Neutral")
plt.plot(featureProvider.smooth(meanPositive, 7)[:120],'g', label="Positive")
plt.plot(featureProvider.smooth(meanNegative, 7)[:120], 'r', label="Negative")
plt.xlabel("time (1/60 s)")
plt.ylabel("mean pupil diameter change (mm)")
plt.legend()


positiveNeutral = []
for i in range(len(meanNeutral)):
	positiveNeutral.append(meanPositive[i] - meanNeutral[i])

negativeNeutral = []
for i in range(len(meanNeutral)):
	negativeNeutral.append(meanNegative[i] - meanNeutral[i])

positiveNegative = []
for i in range(len(meanPositive)):
	positiveNegative.append(meanPositive[i] - meanNegative[i])
plt.figure("Mean difference")
plt.plot(positiveNeutral, label="Positive - Neutral")
plt.plot(negativeNeutral, label="Negative - Neutral")
plt.plot(positiveNegative, label="Positive - Negative")
plt.legend()

for dt in negativeData:
	baselineMean = sum(dt["baselineList"][-30:])/30
	normalized = [pupilDiameter - baselineMean for pupilDiameter in dt["pupilList"]]
	transformed = []

	# bufferList = []
	# MAX_BUFFER = 30
	# for val in normalized:
	# 	if len(bufferList) == 0:
	# 		transformed.append(val)
	# 	else:
	# 		mean = sum(bufferList)/len(bufferList)
	# 		transformed.append(val - mean)

	# 	bufferList.append(val)
	# 	if len(bufferList) == MAX_BUFFER:
	# 		del bufferList[0]

	for val in normalized:
		transformed.append(math.exp(val))

	smoothed = featureProvider.smooth(transformed, 6)
	# plt.plot(smoothed)
plt.show()

# print(sorted(sdVars))



