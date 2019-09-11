class LongShortTermMemory:
	
	def __init__(self, trainFeatures, trainLabels):
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels

	def printEvaluation(self):
		print("Report : Long Short Term Memory")
		print("feature shape : ", self.trainFeatures.shape, ", label shape : ", self.trainLabels.shape)
		