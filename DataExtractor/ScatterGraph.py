from Experiment import Experiment
import numpy as np
from scipy.fftpack import fft
from MovingWindowMean import MovingWindowMean
from MovingWindowFFT import MovingWindowFFT
from SlopeComparison import SlopeComparison
from FFT import FFT

class ScatterGraph(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, csvReader, startIndex, length):
		super(ScatterGraph, self).__init__(csvReader)
		self.name = "Scatter Graph - mean by fft"
		self.startIndex = startIndex
		self.length = length
		# self.meanExp = MovingWindowMean(csvReader)
		# self.fftExp = MovingWindowFFT(csvReader)
		# self.slopeExp = SlopeComparison(csvReader)
		self.fft = FFT()

		# self.N = (end_time - start_time)*sampling_rate # array size


	def process(self, dataList):
		resultX = []
		resultY = []
		for data in dataList:
			target = data[self.startIndex:self.startIndex + self.length]
			mean = sum(target) / len(target)
			fftPeakFreq = self.fft.calculateTimeAtPeakFrequency(target)
			print("mean ", mean, " peak frequency ", fftPeakFreq)
			resultX.append(mean)
			resultY.append(fftPeakFreq)


		return (resultX, resultY)

	def plotData(self):
		self.plotDataMultiScatter()