from Experiment import Experiment

class MeanPlot(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, csvReader):
		super(MeanPlot, self).__init__(csvReader)
		self.name = "Mean Comparison"

	def process(self, dataList):
		result = []
		for data in dataList:
			result.append(sum(data) / len(data))
		return result[:150]

	def plotData(self):
		self.plotDataSingle()
		