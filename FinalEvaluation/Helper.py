import numpy as np
from scipy  import stats
import scipy.interpolate as si
from os import listdir
from os import makedirs
from os import path
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Helper:
	@staticmethod
	def getTrainableData(data, start, window, classCount=3, key="pupilListSmoothed"):
		features = np.random.random((len(data),window,1))
		labels = np.random.random((len(data),classCount))

		for i in range(len(data)):
			for j in range(window):
				features[i][j][0] = data[i][key][start + j]
			labels[i] = Helper.getLabel(data[i]["type"], classCount)
		# print(np.array(features).shape)
		# return np.array(features), np.array(labels)
		# features = np.reshape(features, (features.shape[0],1,features.shape[1]))
		# labels = np.reshape(labels,(labels.shape[0],1,labels.shape[1]))
		# return np.reshape(features, (features.shape[0],1,features.shape[1])), np.reshape(labels,(labels.shape[0],1,labels.shape[1]))
		return features, labels

	@staticmethod
	def getTrainableDataFCNN(data, start, window, classCount=3, key="pupilListSmoothed"):

		features = np.random.random((len(data),9))
		labels = np.random.random((len(data),classCount))

		for i in range(len(data)):
			
			# dt = [d - self.data[i]["baselineMean"] for d in self.data[i]["pupilListSmoothed"][start:start+window]]
			dt = [d for d in data[i][key][start:start+window]]
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

			labels[i] = Helper.getLabel(data[i]["type"], classCount)

		return features, labels

	@staticmethod
	def getTrainableDataFCNNReducedFeature(data, start, window, classCount=3, key="pupilListSmoothed"):

		features = np.random.random((len(data),1,3))
		labels = np.random.random((len(data),classCount))

		for i in range(len(data)):
			
			# dt = [d - self.data[i]["baselineMean"] for d in self.data[i]["pupilListSmoothed"][start:start+window]]
			dt = [d for d in data[i][key][start:start+window]]
			features[i][0][0] = np.std(dt)
			features[i][0][1] = stats.mode(dt)[0][0]
			features[i][0][2] = stats.skew(dt)


			labels[i] = Helper.getLabel(data[i]["type"], classCount)

		return features, labels

	@staticmethod
	def getTrainingDataSlidingWindowMlp(data, dataLength, sampleCount, classCount=3, overlap=0, key="pupilListSmoothed"):
		features = np.random.random((len(data), sampleCount, 3))
		labels = np.random.random((len(data),classCount))

		defaultWindowSize = int(dataLength / sampleCount)
		overlap = min(overlap, .5)
		window = defaultWindowSize + defaultWindowSize * overlap

		for i in range(sampleCount):
			start = max(i * defaultWindowSize - defaultWindowSize * sampleCount, 0)
			
			for j in range(len(data)):
				dt = [d for d in data[j][key][start:start+window]]
				features[j][i][0] = np.std(dt)
				features[j][i][1] = stats.mode(dt)[0][0]
				features[j][i][2] = stats.skew(dt)

		for i in range(len(data)):
			labels[i] = Helper.getLabel(data[i]["type"], classCount)

		return features, labels



	@staticmethod
	def getLabel(label, classCount):
		if classCount == 3:
			if label == 0:
				return [1,0,0]
			if label == 1:
				return [0,1,0]
			if label == 2:
				return [0,0,1]
			if label == 3:
				return [0,0,1]
		elif classCount == 4:
			if label == 0:
				return [1,0,0,0]
			if label == 1:
				return [0,1,0,0]
			if label == 2:
				return [0,0,1,0]
			if label == 3:
				return [0,0,0,1]
		elif classCount == 2:
			if label == 0:
				return [1,0]
			elif label == 1:
				return [0,1]


	@staticmethod
	def getClass(label):
		if label == [1,0,0]:
			return 0
		elif label == [0,1,0]:
			return 1
		else:
			return 2

	def plotConfusionMatrices(confusionMatrices, classes, normalize=False,cmap=plt.cm.Blues, fileName="", title=""):
		fig, axes = plt.subplots(4,3, figsize=(5.5,7.5))
		plt.subplots_adjust(left=.12, bottom=.14, right=.90, top=.90, wspace=.4, hspace=.6)
		# fig.suptitle(title, fontsize=14)
		for i in range(9):
			cm = confusionMatrices[i]
			ax = axes[int(i/3)][(i%3)]
			im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

			ax.set(xticks=np.arange(cm.shape[1]),
						 yticks=np.arange(cm.shape[0]),
						 # ... and label them with the respective list entries
						 title="Participant " + str(i+1),
						 xticklabels=classes, yticklabels=classes,
						 ylabel='True label',
						 xlabel='Predicted label')
			ax.tick_params(
			    axis='x',          # changes apply to the x-axis
			    which='both',      # both major and minor ticks are affected
			    bottom=False,      # ticks along the bottom edge are off
			    top=False,         # ticks along the top edge are off
			    labelbottom=False, labelleft=False) # labels along the bottom edge are off
			ax.tick_params(
			    axis='y',          # changes apply to the x-axis
			    which='both',      # both major and minor ticks are affected
			    bottom=False,      # ticks along the bottom edge are off
			    top=False,         # ticks along the top edge are off
			    labelbottom=False,labelleft=False) # labels along the bottom edge are off
			plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
							 rotation_mode="anchor")
			
			fmt = '.2f' if normalize else 'd'
			thresh = cm.max() / 1.5
			for i in range(cm.shape[0]):
					for j in range(cm.shape[1]):
							ax.text(j, i, format(cm[i, j], fmt),
											ha="center", va="center",
		
											color="white" if cm[i, j] > thresh else "black")
		# fig.tight_layout()
		for i in range(9,12):
			ax = axes[int(i/3)][(i%3)]
			ax.axis('off')

		cm = confusionMatrices[9]
		ax = axes[3][1]
		ax.axis('on')
		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.set(xticks=np.arange(cm.shape[1]),
					 yticks=np.arange(cm.shape[0]),
					 # ... and label them with the respective list entries
					 title="Participant " + str(i+1),
					 xticklabels=classes, yticklabels=classes,
					 ylabel='True label',
					 xlabel='Predicted label')
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
						 rotation_mode="anchor")
		
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 1.5
		for i in range(cm.shape[0]):
				for j in range(cm.shape[1]):
						ax.text(j, i, format(cm[i, j], fmt),
										ha="center", va="center",
		
										color="white" if cm[i, j] > thresh else "black")

		if fileName != "":
			filePath = "images/" + fileName + ".png"
			plt.savefig(filePath, dpi=160)

	def getTitleByDataType(typeId):
		if typeId == "raw":
			return "Raw Data"
		elif typeId == "augmented":
			return "Augmented Data"
		elif typeId == "augmented_train":
			return "Augmented Train Data"

	def plotGlobalConfusionMatrices(confusionMatrices, classes, normalize=False,cmap=plt.cm.Blues, fileName=""):
		fig, axes = plt.subplots(1,3, figsize=(9,3))
		plt.subplots_adjust(left=.12, bottom=.2, right=.96, top=1.0, wspace=1.0, hspace=47)
		
		for i in range(3):
			cm = confusionMatrices[i]["matrix"]
			ax = axes[i]
			im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

			ax.set(xticks=np.arange(cm.shape[1]),
						 yticks=np.arange(cm.shape[0]),
						 # ... and label them with the respective list entries
						 title=Helper.getTitleByDataType(confusionMatrices[i]["dataType"]),
						 xticklabels=classes, yticklabels=classes,
						 ylabel='True label',
						 xlabel='Predicted label')
			# ax.tick_params(
			#     axis='x',          # changes apply to the x-axis
			#     which='both',      # both major and minor ticks are affected
			#     bottom=False,      # ticks along the bottom edge are off
			#     top=False,         # ticks along the top edge are off
			#     labelbottom=False, labelleft=False) # labels along the bottom edge are off
			# ax.tick_params(
			#     axis='y',          # changes apply to the x-axis
			#     which='both',      # both major and minor ticks are affected
			#     bottom=False,      # ticks along the bottom edge are off
			#     top=False,         # ticks along the top edge are off
			#     labelbottom=False,labelleft=False) # labels along the bottom edge are off
			plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
							 rotation_mode="anchor")
			
			fmt = '.2f' if normalize else 'd'
			thresh = cm.max() / 1.25
			for i in range(cm.shape[0]):
					for j in range(cm.shape[1]):
							ax.text(j, i, format(cm[i, j], fmt),
											ha="center", va="center",
		
											color="white" if cm[i, j] > thresh else "black")

		if fileName != "":
			filePath = "images/globals/" + fileName + ".png"
			plt.savefig(filePath, dpi=160)



	def plot_confusion_matrix(confusionMatrix, classes,normalize=False,title=None,cmap=plt.cm.Blues,start=0,window=120,fileName="",modelName=""):
			
		if not title:
				if normalize:
						title = 'Normalized confusion matrix '+modelName+': start - ' + str(start) + ", window - " + str(window)
				else:
						title = 'Confusion matrix '+modelName+': start - ' + str(start) + ", window - " + str(window)

		cm = confusionMatrix

		if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
		else:
				print('Confusion matrix, without normalization')

		print(cm)

		fig, ax = plt.subplots()
		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.figure.colorbar(im, ax=ax)
		# We want to show all ticks...
		ax.set(xticks=np.arange(cm.shape[1]),
					 yticks=np.arange(cm.shape[0]),
					 # ... and label them with the respective list entries
					 xticklabels=classes, yticklabels=classes,
					 title=title,
					 ylabel='True label',
					 xlabel='Predicted label')

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
						 rotation_mode="anchor")

		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
				for j in range(cm.shape[1]):
						ax.text(j, i, format(cm[i, j], fmt),
										ha="center", va="center",
										color="white" if cm[i, j] > thresh else "black")
		fig.tight_layout()
		if fileName != "":
			plt.savefig(fileName, dpi=300)
		# plt.show()
		return ax

	@staticmethod
	def createNewFolder(root):

		folders = [int(f.split('.')[0]) for f in listdir(root) if not isfile(join(root, f))]
		if len(folders) == 0:
			outputFolderName = "0"
		else:
			outputFolderName = str(max(folders) + 1)
		print("Creating folder for output...")
		makedirs(root + "/" + outputFolderName)
		print("Folder creation successful. Output folder name : ", (root+"/" + outputFolderName + "/"))
		return root + "/" + outputFolderName

	@staticmethod
	def createNewFolderNamed(root, outputFolderName):
		if not path.exists(root):
			print("creating ", root)
			makedirs(root)
		folders = [f.split('.')[0] for f in listdir(root) if not isfile(join(root, f))]
		if outputFolderName in folders:
			print("folder already exists")
		else:
			print("Creating folder for output...")
			makedirs(root + "/" + outputFolderName)
			print("Folder creation successful. Output folder name : ", (root+"/" + outputFolderName + "/"))
		return root + "/" + outputFolderName


	@staticmethod
	def bspline(self, cv, n=100, degree=3, periodic=False):
		""" Calculate n samples on a bspline

			cv :      Array ov control vertices
			n  :      Number of samples to return
			degree:   Curve degree
			periodic: True - Curve is closed
					  False - Curve is open
		"""

		# If periodic, extend the point array by count+degree+1
		cv = np.asarray(cv)
		count = len(cv)

		if periodic:
			factor, fraction = divmod(count+degree+1, count)
			cv = np.concatenate((cv,) * factor + (cv[:fraction],))
			count = len(cv)
			degree = np.clip(degree,1,degree)

		# If opened, prevent degree from exceeding count-1
		else:
			degree = np.clip(degree,1,count-1)


		# Calculate knot vector
		kv = None
		if periodic:
			kv = np.arange(0-degree,count+degree+degree-1)
		else:
			kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

		# Calculate query range
		u = np.linspace(periodic,(count-degree),n)


		# Calculate result
		return np.array(si.splev(u, (kv,cv.T,degree))).T

	@staticmethod
	def replaceInvalidByMinimumPupilSize(data, minimumPupilSize):
		for i in range(len(data)):
			if data[i] < minimumPupilSize:
				data[i] = minimumPupilSize

		return data

	@staticmethod
	def getSlidingWindowSkewness(data, window, step):
		i = 0
		n = 0
		result = []
		while i + window < len(data):
			skewness = stats.skew(data[i:i+window])
			result.append(skewness)
			i = i + step
		return result

	@staticmethod
	def getSlidingWindowKurtosis(data, window, step):
		i = 0
		n = 0
		result = []
		while i + window < len(data):
			kurt = stats.kurtosis(data[i:i+window])
			result.append(kurt)
			i = i + step
		return result

	@staticmethod
	def getCentralBaselineMean(data):
		centralArea = data[60:-60]
		mean = sum(centralArea)/len(centralArea)
		return mean

	@staticmethod
	def getFirstMinimumValueAndIndex(data):
		minimum = min(data)
		minIndex = np.where(np.array(data) == minimum)[0][0]
		return minimum, minIndex


	@staticmethod
	def smooth(dataSet, window):
		result = []
		hand = int((window-1)/2)
		for i in range(len(dataSet)):
			if i < hand:
				mean = sum(dataSet[:i+hand+1])/(i+hand+1) 
			elif i + hand >= len(dataSet):
				mean = sum(dataSet[i-hand:])/(hand + len(dataSet) - i)
			else:
				mean = sum(dataSet[i-hand:i+hand+1])/window
			result.append(mean)
		return result

	@staticmethod
	def getWholeSkewness(dataSet, window):
		result = []
		hand = int((window-1)/2)
		for i in range(len(dataSet)):
			if i < hand:
				skew = stats.skew(dataSet[:i+hand+1])
			elif i + hand >= len(dataSet):
				skew = stats.skew(dataSet[i-hand:])
			else:
				skew = stats.skew(dataSet[i-hand:i+hand+1])
			result.append(skew)
		return result

	@staticmethod
	def getWholeKurtosis(dataSet, window):
		result = []
		hand = int((window-1)/2)
		for i in range(len(dataSet)):
			if i < hand:
				kurt = stats.kurtosis(dataSet[:i+hand+1])
			elif i + hand >= len(dataSet):
				kurt = stats.kurtosis(dataSet[i-hand:])
			else:
				kurt = stats.kurtosis(dataSet[i-hand:i+hand+1])
			result.append(kurt)
		return result

	@staticmethod
	def extendZeroZone(pplList, width):
		# for dt in data:
		# pplList = dt['pupilList']
		isZeroZone = pplList[0] == 0

		for i in range(len(pplList)):
			if not isZeroZone and pplList[i] == 0:
				isZeroZone = True
				for j in range(i-width,i):
					if j >= 0:
						pplList[j] = 0
			# if isZeroZone and pplList[i] != 0:
			# 	isZeroZone = False
			# 	for j in range(i + 1, i + width + 1):
			# 		if j < len(pplList):
			# 			pplList[j] = 0
			# 	i = i + width
		# dt['pupilList'] = pplList
		return pplList

	@staticmethod
	def replaceBlankByLinearInterpolation(data):
		# print(data[0], " cccccc")
		x1 = data[0]
		blankZone = x1 == 0
		index = 0
		for i in range(1,len(data)):
			x = data[i]
			log = ""
			if x1 != 0 and x == 0:
				log = log + " blank zone " + str(x1)
				blankZone = True
			if not blankZone and x != 0:
				x1 = x
				index = i
				log = log + " index " + str(i)

			if blankZone and x != 0:
				log = log + " blankZone stopped " + str(x1)
				blankZone = False
				if data[index] == 0:
					data[index] = x
					# print("preblank stopped at index",i)
					# plt.plot(data)
				diff = x - x1
				length = i - index
				# print("replace started")
				for j in range(index + 1, i + 1):
					if j < len(data):
						newVal = x1 + (diff / length) * (j - index)
						# print("Replacing ",j, " with ", newVal)
						data[j] = newVal
					

				x1 = x

			# print(x, log)
		if blankZone:
			for j in range(index + 1, len(data)):
				newVal = x1
				# print("Replacing ",j, " with ", newVal)
				data[j] = newVal

		return data
		