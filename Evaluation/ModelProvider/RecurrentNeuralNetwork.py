class RecurrentNeuralNetwork:
	
	def __init__(self, trainFeatures, trainLabels):
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels

	def printEvaluation(self):
		print("Report : Recurrent Neural Network")
		print("feature shape : ", self.trainFeatures.shape, ", label shape : ", self.trainLabels.shape)
		