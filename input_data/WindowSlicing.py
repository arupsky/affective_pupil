import json
from os import makedirs

frames = [30, 45, 60, 75, 90, 105, 120, 135, 150]
# input_file_name = "raw.json"
# output_folder = "sliced/raw"
input_file_name = "augmented.json"
output_folder = "sliced/augmented"

makedirs(output_folder)
with open(input_file_name) as file:
	file_data = json.load(file)

	for frame in frames:
		output_file_name = output_folder + "/" + str(frame)+".json"
		data = []

		for trial in file_data:
			temp = {}
			# temp["baselineList"] = trial["baselineList"]
			temp["type"] = trial["type"]
			temp["pupilListSmoothed"] = trial["pupilListSmoothed"][:frame]
			data.append(temp)

		with open(output_file_name, 'w') as out_file:
			json.dump(data, out_file)

