from scipy.fftpack import fft
import numpy as np

class FFT:
	def __init__(self):
		self.sampling_rate = 1000 # Hz

	def calculatePeakFrequency(self, data):
		windowSize = len(data)
		N = windowSize;
		result = []
		T = 1/self.sampling_rate # inverse of the sampling rate
		x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
		yr = fft(data)
		y = 2/N * np.abs(yr[0:np.int(N/2)])
		maxY = max(y)
		index = np.where(y == [maxY])
		return index[0]

	def calculateTimeAtPeakFrequency(self, data):
		windowSize = len(data)
		N = windowSize;
		result = []
		T = 1/self.sampling_rate # inverse of the sampling rate
		x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
		yr = fft(data)
		y = 2/N * np.abs(yr[0:np.int(N/2)])
		maxY = max(y)
		return maxY

