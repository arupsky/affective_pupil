from Experiment import Experiment

class ChainedExperiment(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, name, csvReader, experimentChain):
		super(ChainedExperiment, self).__init__(csvReader)
		self.name = name
		self.experimentChain = experimentChain


	def process(self, dataList):
		result = dataList
		for experiment in self.experimentChain:
			result = experiment.process(result)
		return result

	def plotData(self):
		self.plotDataMulti()