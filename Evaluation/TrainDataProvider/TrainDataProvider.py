import numpy as np
from Helper import Helper
from scipy  import stats

class TrainDataProvider:
	"""docstring for TrainDataProvider"""
	def __init__(self, formattedData):
		self.data = formattedData
		self.classCount = 3

	def getTrainableData(self, start, window):
		# labels = []
		# features = []
		features = np.random.random((len(self.data),window,1))
		labels = np.random.random((len(self.data),self.classCount))
		# for dt in self.data:
		# 	pupilData = np.array(dt["pupilList"][start:start+window])
		# 	pupilData = pupilData - dt["baselineMean"]
			
		# 	pupilData = pupilData.reshape(pupilData.shape[0],1)
		# 	# print(pupilData)
		# 	features.append(pupilData)
		# 	labels.append(Helper.getLabel(dt["type"], 3))

		for i in range(len(self.data)):
			# print(len(self.data[i]["pupilList"]))
			for j in range(window):
				# print(start + j)
				features[i][j][0] = self.data[i]["pupilList"][start + j] - self.data[i]["baselineMean"]
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
			
			dt = [d - self.data[i]["baselineMean"] for d in self.data[i]["pupilList"][start:start+window]]
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


		