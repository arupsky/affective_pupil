from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
from sklearn.preprocessing import Imputer
import pandas as pd
from Helper import Helper

class TsvParser:
	"""docstring for TsvParser"""
	def __init__(self, folderName, globalConfig = {}):
		self.folderName = folderName
		self.minimumPupilSize = 1.4
		self.globalConfig = globalConfig

	def getFileNames(self):
		fileNames = [f for f in listdir(self.folderName) if (isfile(join(self.folderName, f)) and ".tsv" in f)]
		return fileNames

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

	def replaceBySplineBase(self, data, degree, base):
		controlPoints = []
		segmentLength = int(300/base)
		segment = 0
		while segment * segmentLength < len(data):
			controlPoints.append([segment, data[segment * segmentLength]])
			segment = segment + 1
		if (segment - 1) * segmentLength != len(data) - 1:
			controlPoints.append([segment, data[len(data) - 1]])

		cv = np.array(controlPoints)
		p = self.bspline(cv,n=len(data),degree=degree,periodic=False)
		x,y = p.T
		return y

	def replaceBySpline(self, data, degree):
		controlPoints = []
		for i in range(len(data)):
			controlPoints.append([i,data[i]])
		cv = np.array(controlPoints)
		p = self.bspline(cv,n=len(data),degree=degree,periodic=False)
		x,y = p.T
		return y

	def correction(self, data, do_plot = False):
		# data = Helper.extendZeroZone(data, 2)

		if do_plot:
			self.correction2(data)
			# fig = plt.figure()
			# ax = fig.add_subplot(311)
			# ax.plot(data, label="raw")

		data = Helper.replaceBlankByLinearInterpolation(data)
		# if do_plot:
		# 	ax = fig.add_subplot(312)
		# 	ax.plot(data, label="replacing blank")
		data = Helper.replaceInvalidByMinimumPupilSize(data, self.minimumPupilSize)
		data = Helper.smooth(data, 5)
		data = Helper.smooth(data, 5)
		data = Helper.smooth(data, 7)
		data = Helper.smooth(data, 7)
		# if do_plot:
		# 	ax = fig.add_subplot(313)
		# 	ax.plot(data, label="smoothed")
		return data

	def correction2(self, data, do_plot = False):
		
		if do_plot:
			fig = plt.figure()
			ax = fig.add_subplot(311)
			# ax.set_title("Raw data")
			ax.plot(data, 'r', label='Raw data')
			plt.legend()
		dt = Helper.smooth(data, 5)
		dt = self.speedOutlierDetection(dt, 10)
		# if do_plot:
		# 	ax = fig.add_subplot(512)
		# 	ax.set_title("5 point smoothing + outlier (velocity) marking")
		# 	ax.plot(dt, 'g')
		dt = Helper.replaceBlankByLinearInterpolation(dt)
		dt = Helper.smooth(dt, 5)
		if do_plot:
			ax = fig.add_subplot(312)
			# ax.set_title("Outlier detection + Liner interpolation + smoothing")
			ax.plot(dt, 'b', label='Processed')
			plt.legend()
		dt = self.speedOutlierDetection(dt, 5)
		dt = Helper.replaceBlankByLinearInterpolation(dt)
		dt = Helper.smooth(dt, 5)
		dt = Helper.smooth(dt, 7)
		dt = Helper.smooth(dt, 7)
		
		if do_plot:
			ax = fig.add_subplot(313)
			# ax.set_title("Final")
			ax.plot(dt, 'g', label='Final')
			plt.legend()


		return dt
		

	def transformWithThreshold(self, x, threshold):
		if x < threshold:
			return 1
		else:
			return 0

	def speedOutlierDetection(self, data, n):
		speedLine = []
		# plt.figure()

		for i in range(1, len(data) - 1):
			speedLine.append(max(abs(data[i] - data[i - 1]), abs(data[i + 1] - data[i])))

		# plt.plot(data, 'r')
		median = np.median(speedLine)
		MAD = np.median([abs(d - median) for d in speedLine])
		threshold = median + n * MAD

		newData = [data[0]]

		for i in range(len(speedLine)):
			if self.transformWithThreshold(speedLine[i], threshold):
				newData.append(data[i + 1])
			else:
				newData.append(0)

		# newData = [self.transformWithThreshold(x, threshold) for x in speedLine]
		# plt.plot(newData, 'g')
		return newData


	def getNewTrialData(self, typeIndex):
		trialData = {}
		trialData["type"] = int(typeIndex)
		trialData["baselineList"] = []
		trialData["baselineTime"] = []
		trialData["baselineMean"] = 0
		trialData["pupilList"] = []
		trialData["pupilTime"] = []
		trialData["pupilMean"] = 0
		return trialData

	def insertInTrialData(self, isBaseline, trialData, parts):
		leftValid = (parts[6] == "Valid")
		rightValid = (parts[7] == "Valid")

		if leftValid:
			left = float(parts[4])

		if rightValid:
			right = float(parts[5])

		if leftValid:
			if isBaseline:
				trialData["baselineList"].append(left)
				trialData["baselineTime"].append(int(parts[0]))
			else:
				trialData["pupilList"].append(left)
				trialData["pupilTime"].append(int(parts[0]))

		elif rightValid:
			if isBaseline:
				trialData["baselineList"].append(right)
				trialData["baselineTime"].append(int(parts[0]))
			else:
				trialData["pupilList"].append(right)
				trialData["pupilTime"].append(int(parts[0]))

		else:
			if isBaseline:
				trialData["baselineList"].append(0)
				trialData["baselineTime"].append(int(parts[0]))
			else:
				trialData["pupilList"].append(0)
				trialData["pupilTime"].append(int(parts[0]))

	def hasTooManyInvalidData(self, trialData):
		invalid = sum([1 for x in trialData['pupilList'][:300] if x == 0])
		if invalid > 70:
			return True

		validBaseline = [x for x in trialData['baselineList'][-30:] if x != 0]
		if len(validBaseline) < 10:
			return True

		return False

	def processFile(self,fileName):
		with open(self.folderName + "/" + fileName) as f:
		    content = f.readlines()
		content = [x.strip() for x in content if ".bmp" in x] 
		
		lastStimuliName = ""
		counter = 0
		experimentData = []
		trialData = {}
		isBaseline = False

		doPrint = False

		for line in content:
			
			parts = line.split('\t')

			if lastStimuliName != parts[8]: # found new stimuli name
				lastStimuliName = parts[8]
				isBaseline = not isBaseline
				
				if counter%2 == 0: # found data for new trial
					p = lastStimuliName.split('_')
					trialData = self.getNewTrialData(p[1])
					experimentData.append(trialData)

				counter = counter + 1

			self.insertInTrialData(isBaseline, trialData, parts)
				

		# cnt = 0
		# # debugCnt = list(range(len(experimentData)))
		# debugCnt = list(range(self.globalConfig["preprocess_graph_count_per_file"]))
		# # debugCnt = [2]


		# blackList = []
		# for trialData in experimentData:
		# 	if self.globalConfig["debug_data_collect"]:
		# 		print("--------------------------------------------", cnt)

		# 	if self.hasTooManyInvalidData(trialData):
		# 		blackList.append(trialData)
		# 		if self.globalConfig["debug_data_collect"]:
		# 			print("### data ", str(cnt), " added to blacklist due to too many invalid data")
		# 		continue

		# 	# validBaseline = [x for x in trialData['baselineList'][-30:] if x != 0]
		# 	# baselineMean = sum(validBaseline)/len(validBaseline)

		# 	# if cnt in debugCnt:
		# 	# 	plt.figure()
		# 	# 	plt.plot(trialData["pupilList"][:300], label='Raw data')
		# 	trialData["baselineTime"] = [x - trialData["baselineTime"][0] for x in trialData["baselineTime"][:300]]
		# 	trialData["pupilTime"] = [x - trialData["pupilTime"][0] for x in trialData["pupilTime"][:300]]
		# 	trialData["pupilListSmoothed"] = self.correction2(trialData["pupilList"][:300], do_plot=self.globalConfig["generate_preprocess_graph"] and (cnt in debugCnt))
		# 	trialData["baselineList"] = self.correction2(trialData["baselineList"][:300], do_plot=self.globalConfig["generate_preprocess_graph"] and (cnt in debugCnt))
		# 	# trialData["pupilMean"] = np.mean(trialData["pupilList"])
		# 	# trialData["baselineMean"] = baselineMean

		# 	# if cnt in debugCnt:
		# 	# 	plt.figure()
		# 	# 	plt.plot(trialData["pupilList"][:300], label='Raw data')
		# 	# 	plt.plot(trialData["pupilListSmoothed"],  label='Processed Data')
		# 	# 	plt.legend()
		# 	cnt = cnt + 1
			

		# for dt in blackList:
		# 	experimentData.remove(dt)

		return experimentData

	def processData(self, experimentData):
		cnt = 0
		# debugCnt = list(range(len(experimentData)))
		debugCnt = list(range(self.globalConfig["preprocess_graph_count_per_file"]))
		# debugCnt = [2]


		blackList = []
		for trialData in experimentData:
			if self.globalConfig["debug_data_collect"]:
				print("--------------------------------------------", cnt)

			if self.hasTooManyInvalidData(trialData):
				blackList.append(trialData)
				if self.globalConfig["debug_data_collect"]:
					print("### data ", str(cnt), " added to blacklist due to too many invalid data")
				continue

			# validBaseline = [x for x in trialData['baselineList'][-30:] if x != 0]
			# baselineMean = sum(validBaseline)/len(validBaseline)

			# if cnt in debugCnt:
			# 	plt.figure()
			# 	plt.plot(trialData["pupilList"][:300], label='Raw data')

			# temp = trialData["baselineList"][]
			# print("baseline length ", len(trialData["baselineList"]), ", pupilList length ", len(trialData["pupilList"])) 

			isShowingPlot = self.globalConfig["generate_preprocess_graph"] and (cnt in debugCnt)
			temp = trialData["baselineList"][-200:]
			temp.extend(trialData["pupilList"][:300])
			corrected = self.correction2(temp, do_plot=isShowingPlot)

			trialData["baselineTime"] = [x - trialData["baselineTime"][0] for x in trialData["baselineTime"][:300]]
			trialData["pupilTime"] = [x - trialData["pupilTime"][0] for x in trialData["pupilTime"][:300]]
			# trialData["pupilListSmoothed"] = self.correction2(trialData["pupilList"][:300], do_plot=self.globalConfig["generate_preprocess_graph"] and (cnt in debugCnt))
			# trialData["baselineList"] = self.correction2(trialData["baselineList"][-300:], do_plot=self.globalConfig["generate_preprocess_graph"] and (cnt in debugCnt))
			trialData["pupilListSmoothed"] = corrected[200:]
			trialData["baselineList"] = corrected[:200]
			if isShowingPlot:
				plt.figure()
				plt.plot(trialData["pupilListSmoothed"])
				plt.plot(trialData["baselineList"])


			# if cnt in debugCnt:
			# 	plt.figure()
			# 	plt.plot(trialData["pupilList"][:300], label='Raw data')
			# 	plt.plot(trialData["pupilListSmoothed"],  label='Processed Data')
			# 	plt.legend()
			cnt = cnt + 1
			

		for dt in blackList:
			experimentData.remove(dt)

		return experimentData

	def loadData(self):
		if self.globalConfig["participant"] < 1:
			return self.loadFolder()
		else:
			return self.loadIndividualData(self.globalConfig["participant"])

	def loadFolder(self):
		fileNames = self.getFileNames()
		data = []
		for fileName in fileNames:
			data.extend(self.processFile(fileName))
		return data

	def loadIndividualData(self, participant):
		fileNames = self.getFileNames()
		data = []
		for fileName in fileNames:
			if "p" + str(participant) + ".tsv" in fileName:
				data.extend(self.processFile(fileName))
				break
		return data

	def generateSampleFigures(self):
		pass
		
