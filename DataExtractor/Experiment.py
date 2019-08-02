import matplotlib.pyplot as plt

class Experiment:
	"""docstring for Experiment"""
	def __init__(self, csvParser):
		super(Experiment, self).__init__()
		self.csvParser = csvParser
		self.name = ""
		# self.key = "zeroBasedPupilList"
		self.key = "normalizedPupilList"

	def plotDataMulti(self):
		neg1 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData1(),self.key));
		neg2 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData2(),self.key));
		pos = self.process(self.csvParser.getDataForKey(self.csvParser.getPositiveData(),self.key));
		neu = self.process(self.csvParser.getDataForKey(self.csvParser.getNeutralData(),self.key));

		fig = plt.figure(self.name, figsize=(10,8))
		fig.suptitle(self.name, fontsize=16)

		neutralPlot = plt.subplot(221)
		neutralPlot.set_title("Neutral Stimuli")
		for dt in neu:
			neutralPlot.plot(dt)

		positivePlot = plt.subplot(222)
		positivePlot.set_title("Positive Stimuli")
		for dt in pos:
			positivePlot.plot(dt)

		negativePlot1 = plt.subplot(223)
		negativePlot1.set_title("Negative Stimuli 1")
		for dt in neg1:
			negativePlot1.plot(dt)

		negativePlot2 = plt.subplot(224)
		negativePlot2.set_title("Negative Stimuli 2")
		for dt in neg2:
			negativePlot2.plot(dt)
		# plt.show()

	def plotDataMultiXY(self):
		neg1 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData1(),self.key));
		neg2 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData2(),self.key));
		pos = self.process(self.csvParser.getDataForKey(self.csvParser.getPositiveData(),self.key));
		neu = self.process(self.csvParser.getDataForKey(self.csvParser.getNeutralData(),self.key));

		fig = plt.figure(self.name, figsize=(10,8))
		fig.suptitle(self.name, fontsize=16)

		neutralPlot = plt.subplot(221)
		neutralPlot.set_title("Neutral Stimuli")
		for dt in neu:
			neutralPlot.plot(dt[0], dt[1])

		positivePlot = plt.subplot(222)
		positivePlot.set_title("Positive Stimuli")
		for dt in pos:
			positivePlot.plot(dt[0], dt[1])

		negativePlot1 = plt.subplot(223)
		negativePlot1.set_title("Negative Stimuli 1")
		for dt in neg1:
			negativePlot1.plot(dt[0], dt[1])

		negativePlot2 = plt.subplot(224)
		negativePlot2.set_title("Negative Stimuli 2")
		for dt in neg2:
			negativePlot2.plot(dt[0], dt[1])

	def plotDataMultiScatter(self):
		neg1 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData1(),self.key));
		neg2 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData2(),self.key));
		pos = self.process(self.csvParser.getDataForKey(self.csvParser.getPositiveData(),self.key));
		neu = self.process(self.csvParser.getDataForKey(self.csvParser.getNeutralData(),self.key));

		fig = plt.figure(self.name, figsize=(10,8))
		fig.suptitle(self.name, fontsize=16)

		ax = fig.subplots()

		ax.scatter(neg1[0], neg1[1], c='red')
		ax.scatter(neg2[0], neg2[1], c='red')
		ax.scatter(neu[0], neu[1], c='green')
		ax.scatter(pos[0], pos[1], c='blue')

		# neutralPlot = plt.subplot(221)
		# neutralPlot.set_title("Neutral Stimuli")
		# for dt in neu:
		# 	neutralPlot.plot(dt[0], dt[1])

		# positivePlot = plt.subplot(222)
		# positivePlot.set_title("Positive Stimuli")
		# for dt in pos:
		# 	positivePlot.plot(dt[0], dt[1])

		# negativePlot1 = plt.subplot(223)
		# negativePlot1.set_title("Negative Stimuli 1")
		# for dt in neg1:
		# 	negativePlot1.plot(dt[0], dt[1])

		# negativePlot2 = plt.subplot(224)
		# negativePlot2.set_title("Negative Stimuli 2")
		# for dt in neg2:
		# 	negativePlot2.plot(dt[0], dt[1])

	def plotDataSingle(self):
		neg1 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData1(),self.key));
		neg2 = self.process(self.csvParser.getDataForKey(self.csvParser.getNegativeData2(),self.key));
		pos = self.process(self.csvParser.getDataForKey(self.csvParser.getPositiveData(),self.key));
		neu = self.process(self.csvParser.getDataForKey(self.csvParser.getNeutralData(),self.key));

		fig = plt.figure(self.name, figsize=(10,8))
		fig.suptitle(self.name, fontsize=16)
		plt.subplot(221).set_title("Neutral Stimuli")
		# for dt in neu:
		plt.plot(neu)
		plt.subplot(222).set_title("Positive Stimuli")
		# for dt in pos:
		plt.plot(pos)
		plt.subplot(223).set_title("Negative Stimuli 1")
		# for dt in neg1:
		plt.plot(neg1)
		plt.subplot(224).set_title("Negative Stimuli 2")
		# for dt in neg2:
		plt.plot(neg2)
		# plt.show()


	def process(self, dataList):
		raise NotImplementedError("Please Implement this method")

	def plotData(self, dataList):
		raise NotImplementedError("Please Implement this method")


		