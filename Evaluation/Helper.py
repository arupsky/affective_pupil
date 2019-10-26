class Helper:
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


	@staticmethod
	def replaceInvalidByMinimumPupilSize(data, minimumPupilSize):
		for i in range(len(data)):
			if data[i] < minimumPupilSize:
				data[i] = minimumPupilSize

		return data


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
		