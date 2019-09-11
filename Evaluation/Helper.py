class Helper:
	@staticmethod
	def getLabel(label, classCount):
		if classCount == 3:
			if label == 0:
				return [1,0,0]
			if label == 1:
				return [1,0,0]
			if label == 2:
				return [0,1,0]
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
		