import json
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si

# input_file_name = "raw.json"
# output_file_name = "augmented_5000.json"
input_file_name = "validation.json"
output_file_name = "augmented_train_validation.json"
count_per_type = 2000


def bspline(cv, n=100, degree=3, periodic=False):
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

def getBspline(data, length=100, degree=3):
	controlPoints = []
	for i in range(len(data)):
		controlPoints.append([i,data[i]])
	cv = np.array(controlPoints)
	p = bspline(cv,n=length,degree=degree,periodic=False)
	x,y = p.T
	return y

def randomPick(dataset):
	return random.choice(dataset)

def getSyntheticData(pupilList, start = 0, end = 150):
	window_size = int((end - start) * .1)
	window_start = random.choice(range(start, end - window_size))
	scale_factor = .2 + .3 * random.random()
	if random.random() < .5:
		target_window = window_size + int(window_size * scale_factor)
	else:
		target_window = window_size - int(window_size * scale_factor)

	# fig = plt.figure()
	# ax1 = fig.add_subplot(311)
	# ax1.plot(pupilList, label='source')
	# ax1.axvline(x=window_start, color='red', linestyle='--')
	# ax1.axvline(x=(window_start + window_size), color='red', linestyle='--')
	# plt.legend()

	# print("window_size", window_size)
	# print("window_start", window_start)
	# print("target_window", target_window)

	augmentPart = getBspline(pupilList[window_start:window_start + window_size], length=target_window, degree=3)

	newPupilList = pupilList[:window_start]
	newPupilList.extend(augmentPart)
	newPupilList.extend(pupilList[window_start + window_size:])


	# ax2 = fig.add_subplot(312)
	# ax2.plot(newPupilList, label='window warping')
	# ax2.axvline(x=window_start, color='red', linestyle='--')
	# ax2.axvline(x=(window_start + target_window), color='red', linestyle='--')
	# plt.legend()

	smoothed = getBspline(newPupilList, length=len(newPupilList), degree=3)

	# ax3 = fig.add_subplot(313)
	# ax3.plot(smoothed, label='final')
	# plt.legend()


	return list(smoothed)


def generateAugmentedData(data):
	temp = {}
	temp["type"] = data["type"]
	temp["pupilListSmoothed"] = getSyntheticData(data["pupilListSmoothed"])
	return temp

def getBasicStructure(data):
	temp = {}
	temp["type"] = data["type"]
	temp["pupilListSmoothed"] = data["pupilListSmoothed"]
	return temp

with open(input_file_name) as file:
	file_data = json.load(file)
	with open("train.json") as file2:
		file_data.extend(json.load(file2))
	negative_data = [x for x in file_data if x["type"] == 0]
	neutral_data = [x for x in file_data if x["type"] == 1]
	positive_data = [x for x in file_data if x["type"] == 2]

	new_data_set = [getBasicStructure(x) for x in file_data]
	print("Total length", len(new_data_set))
	print("Augmentation started")
	for i in range(count_per_type):
		print(i)
		data = randomPick(negative_data)
		synthetic = generateAugmentedData(data)
		new_data_set.append(synthetic)

		data = randomPick(neutral_data)
		synthetic = generateAugmentedData(data)
		new_data_set.append(synthetic)

		data = randomPick(positive_data)
		synthetic = generateAugmentedData(data)
		new_data_set.append(synthetic)

	print("Augmentation finished.")
	print("Final length", len(new_data_set))

	with open(output_file_name, 'w') as out_file:
		json.dump(new_data_set, out_file)
	# plt.show()
