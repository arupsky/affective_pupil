from CSVParser import CSVParser
import numpy as np
# import numpy
import matplotlib.pyplot as plt
from scipy.fftpack import fft

csvParser = CSVParser()
data = csvParser.getDataForKey(csvParser.getPositiveData(),"zeroBasedPupilList")

samplingRate = 1000
windowSize = 40
N = windowSize
T = 1 / samplingRate
x = np.linspace(0, (1/(2 * T)), int(N/2))

yr = fft(data[0][0:windowSize])
y = 2/N * np.abs(yr[0:np.int(N/2)])
maxY = max(y)
result = np.where(y == [maxY])
# print(len(data[0][0:150]))
print(result[0])
plt.plot(x,y)
plt.show()
