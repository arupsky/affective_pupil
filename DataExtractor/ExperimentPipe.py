from CSVParser import CSVParser
import matplotlib.pyplot as plt
import pandas as pd
# from tsfresh import extract_features
# from tsfresh import select_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh import extract_relevant_features
import numpy as np
from SimpleDataPlot import SimpleDataPlot
from MeanPlot import MeanPlot
from MovingWindowMean import MovingWindowMean
from SlopeComparison import SlopeComparison
from MovingWindowFFT import MovingWindowFFT
from MovingWindowFFTFreq import MovingWindowFFTFreq
from MovingWindowMeanFFT import MovingWindowMeanFFT
from ChainedExperiment import ChainedExperiment
from ScatterGraph import ScatterGraph


def normalizedPupilComparison(csvParser):
	fig = plt.figure("Normalized pupil data comparison", figsize=(10,8))
	fig.suptitle('Normalized pupil data comparison', fontsize=16)
	plt.subplot(221).set_title("Neutral Stimuli")
	plotFirst100Points(csvParser.getNeutralData(), "zeroBasedPupilList", "neutralData")
	plt.subplot(222).set_title("Positive Stimuli")
	plotFirst100Points(csvParser.getPositiveData(), "zeroBasedPupilList", "positiveData")
	plt.subplot(223).set_title("Negative Stimuli 1")
	plotFirst100Points(csvParser.getNegativeData1(), "zeroBasedPupilList", "negativeData1")
	plt.subplot(224).set_title("Negative Stimuli 2")
	plotFirst100Points(csvParser.getNegativeData2(), "zeroBasedPupilList", "negativeData2")
	plt.show()


def meanComparison(csvParser):
	pass

def featureExtraction(pupilData, key, count):
	stimuliList = []
	index = 0
	firstTime = True

	for dt in pupilData:
		print("pupil data ", index)
		d = {'id': index, 'pupil_size': dt[key][:count]}
		df = pd.DataFrame(data=d)
		index = index + 1
		stimuliList.append(df)
	finalDataFrame = stimuliList[0]
	for i in range(1, len(stimuliList)):
		# print(stimuliList[i])
		finalDataFrame = finalDataFrame.append(stimuliList[i], ignore_index=True)
	
	# print(finalDataFrame)
	extracted_features = extract_features(finalDataFrame, column_id="id")

	# y = {'id' : list(range(len(finalDataFrame))), 'size':np.array([True] * 4)}
	# dataY = pd.DataFrame(data = y)

	impute(extracted_features)
	# features_filtered = select_features(extracted_features, dataY)

	print(len(extracted_features))


def plotFirst100Points(pupilData, key, name):
	# f = plt.figure(name)
	for dt in pupilData:
		plt.plot(dt[key][:150])

csvParser = CSVParser()

simpleDataPlot = SimpleDataPlot(csvParser)
simpleDataPlot.plotData()

# movingWindowMean = MovingWindowMean(csvParser)
# movingWindowMean.plotData()

# slopeComparison = SlopeComparison(csvParser)
# slopeComparison.plotData()

# movingWindowFFT = MovingWindowFFT(csvParser)
# movingWindowFFT.plotData()

# movingWindowFFTFreq = MovingWindowFFTFreq(csvParser)
# movingWindowFFTFreq.plotData()

# movingWindowMeanFFT = MovingWindowMeanFFT(csvParser)
# movingWindowMeanFFT.plotData()

# meanFFTFreqSlope = ChainedExperiment("Moving window mean -> FFT -> slope",csvParser, [MovingWindowMean(csvParser), MovingWindowFFTFreq(csvParser), SlopeComparison(csvParser)])
# meanFFTFreqSlope.plotData()


# temp = ChainedExperiment("FFT -> Mean -> FFT", csvParser, [MovingWindowFFT(csvParser)])
# temp.plotData()

# temp = ScatterGraph(csvParser, 60, 180)
# temp.plotData()


plt.show()


# normalizedPupilComparison(csvParser)

# plt.show()



# featureExtraction(csvParser.getNegativeData1(), "zeroBasedPupilList", 150)

# print(csvParser.getNegativeData1())