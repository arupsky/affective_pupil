from os import listdir
from os import makedirs
from os.path import isfile, join
import scipy.interpolate as si
from tsvparser import TsvParser
import random
from Helper import Helper
import math
from Formatter.Formatter import Formatter
import matplotlib.pyplot as plt
from Plotter import Plotter
import json


config = {}


config["output_train_file"] = "../input_data/train_raw.json"
config["output_test_file"] = "../input_data/test_raw.json"


config["minimum_as_baseline"] = False
config["fixation_as_baseline"] = True
config["fixation_minimum_interpolate"] = False
config["generate_preprocess_graph"] = False   # Fertig
config["preprocess_graph_count_per_file"] = 0   # Fertig
config["debug_data_collect"] = False   # Fertig


participants = list(range(1,11)) 
testGroup = random.sample(participants, 3)
trainGroup = [x for x in participants if x not in testGroup]
print(testGroup)
print(trainGroup)

tsvParser = TsvParser("../tobi_data/", globalConfig = config)
formatter = Formatter(config)

trainDataRaw = []
testDataRaw = []

for p in trainGroup:
	data = tsvParser.loadIndividualData(str(p))
	data = tsvParser.processData(data)
	data = formatter.process(data)
	trainDataRaw.extend(data)

for p in testGroup:
	data = tsvParser.loadIndividualData(str(p))
	data = tsvParser.processData(data)
	data = formatter.process(data)
	testDataRaw.extend(data)

print(len(trainDataRaw))
print(len(testDataRaw))
print("Total",len(trainDataRaw) + len(testDataRaw))
# data = tsvParser.loadIndividualData("1")
# print(len(data))


# trainPlotter = Plotter(trainDataRaw)
# testPlotter = Plotter(testDataRaw)

# trainPlotter.plotAll("pupilListSmoothed")
# testPlotter.plotAll("pupilListSmoothed")

# plt.show()

with open(config["output_train_file"], 'w') as outfile:
	json.dump(trainDataRaw, outfile)

with open(config["output_test_file"], 'w') as outfile:
	json.dump(testDataRaw, outfile)