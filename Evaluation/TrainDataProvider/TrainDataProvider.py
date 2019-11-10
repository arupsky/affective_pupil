import numpy as np
from Helper import Helper
from scipy  import stats

class TrainDataProvider:
	"""docstring for TrainDataProvider"""
	def __init__(self, formattedData):
		self.data = formattedData
		self.classCount = 3
		self.key = "pupilListSmoothed"
		# self.key = "pupilKurtosis"
		# self.key = "pupilSkewness"

	def getTrainableDataSlidingWindow(self, start, window, sliding_window = 10, step = 3):

		labels = []
		features = []
		for dt in self.data:
			ppl = dt[self.key][start:start+window]
			feature = []
			i = 0
			
			while i + sliding_window < len(ppl):
				temp = ppl[i:i+sliding_window]
				feature.append(np.mean(temp))
				feature.append(stats.kurtosis(temp))
				feature.append(stats.skew(temp))
				i = i + step
			labels.append(Helper.getLabel(dt["type"], 3))
			features.append(feature)

		features = np.array(features)
		labels = np.array(labels)

		features = features.reshape((features.shape[0], features.shape[1], 1))

		return features, labels




	def getTrainableData(self, start, window):
		features = np.random.random((len(self.data),window,1))
		labels = np.random.random((len(self.data),self.classCount))

		for i in range(len(self.data)):
			for j in range(window):
				features[i][j][0] = self.data[i][self.key][start + j]
			labels[i] = Helper.getLabel(self.data[i]["type"], 3)
		# print(np.array(features).shape)
		# return np.array(features), np.array(labels)
		# features = np.reshape(features, (features.shape[0],1,features.shape[1]))
		# labels = np.reshape(labels,(labels.shape[0],1,labels.shape[1]))
		# return np.reshape(features, (features.shape[0],1,features.shape[1])), np.reshape(labels,(labels.shape[0],1,labels.shape[1]))
		return features, labels

	def getTrainableDataFCNN(self, start, window):

		features = np.random.random((len(self.data),9))
		labels = np.random.random((len(self.data),self.classCount))

		for i in range(len(self.data)):
			
			# dt = [d - self.data[i]["baselineMean"] for d in self.data[i]["pupilListSmoothed"][start:start+window]]
			dt = [d for d in self.data[i][self.key][start:start+window]]
			# print(stats.mode(dt)[0][0])
			features[i][0] = np.mean(dt)
			features[i][1] = np.median(dt)
			features[i][2] = np.std(dt)
			features[i][3] = stats.mode(dt)[0][0]
			features[i][4] = stats.kurtosis(dt)
			features[i][5] = stats.skew(dt)
			features[i][6] = min(dt)
			features[i][7] = max(dt)
			features[i][8] = max(dt) - min(dt)

			labels[i] = Helper.getLabel(self.data[i]["type"], 3)

		return features, labels


		