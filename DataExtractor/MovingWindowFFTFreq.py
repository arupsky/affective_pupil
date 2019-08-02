from Experiment import Experiment
import numpy as np
from scipy.fftpack import fft

class MovingWindowFFTFreq(Experiment):
	"""docstring for MeanPlot"""
	def __init__(self, csvReader):
		super(MovingWindowFFTFreq, self).__init__(csvReader)
		self.name = "Moving Window FFT Comparison (Frequency)"
		# Sampling rate and time vector
		self.start_time = 0 # seconds
		self.end_time = 2 # seconds
		self.sampling_rate = 1000 # Hz
		# self.N = (end_time - start_time)*sampling_rate # array size


	def process(self, dataList):
		windowSize = 40
		N = windowSize;
		result = []
		T = 1/self.sampling_rate # inverse of the sampling rate
		x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
		
		for data in dataList:
			start = 0
			fftValues = []
			while start < len(data[:150]) - windowSize:
				yr = fft(data[start:start + windowSize])
				y = 2/N * np.abs(yr[0:np.int(N/2)])
				maxY = max(y)
				index = np.where(y == [maxY])
				fftValues.append(index[0])
				start = start + 1

			result.append(fftValues)
		return result

	def plotData(self):
		self.plotDataMulti()