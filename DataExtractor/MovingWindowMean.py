from Experiment import Experiment

class MovingWindowMean(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, csvReader):
		super(MovingWindowMean, self).__init__(csvReader)
		self.name = "Moving Window Mean Comparison"

	def process(self, dataList):
		windowSize = 40
		result = []
		
		for data in dataList:
			start = 0
			meanValues = []
			while start < len(data) - windowSize:
				mean = sum(data[start:start + windowSize]) / windowSize
				meanValues.append(mean)
				start = start + 1
				pass
			result.append(meanValues[:150])
		return result

	def plotData(self):
		self.plotDataMulti()