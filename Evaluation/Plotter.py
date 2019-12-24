import matplotlib.pyplot as plt

class Plotter:
	"""docstring for Plotter"""
	def __init__(self, data):
		self.data = data

	def plotAll(self, key):
		plt.figure()
		for dt in self.data:
			plt.plot(dt[key])
		