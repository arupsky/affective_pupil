from Experiment import Experiment

class SimpleDataPlot(Experiment):
	def __init__(self, csvParser):
		super(SimpleDataPlot, self).__init__(csvParser)
		self.name = "Normalized pupil data comparison"

	def process(self, dataList):
		result = []
		for dt in dataList:
			result.append(dt[:60])
		return result

	def plotData(self):
		self.plotDataMulti()