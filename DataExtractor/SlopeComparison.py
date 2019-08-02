from Experiment import Experiment

class SlopeComparison(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, csvReader):
		super(SlopeComparison, self).__init__(csvReader)
		self.name = "Slope Comparison"

	def process(self, dataList):
		result = []
		distance = 3
		for data in dataList:
			start = 0
			slopeValues = []
			while start < len(data) - distance:
				slope = (data[start + distance] - data[start])/distance

				slopeValues.append(slope)
				start = start + 1
				pass
			result.append(slopeValues[:150])
		return result

	def plotData(self):
		self.plotDataMulti()