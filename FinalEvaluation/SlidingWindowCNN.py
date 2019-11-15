from Helper import Helper
from CNN_Re import CNN_Re
import json
import tensorflow as tf
from statistics import mode 
from collections import Counter
import numpy as np

experiment = "SlidingWindowCNN"
# folder = "augmented_5000"
# folder = "augmented"
folder = "raw"
# folder = "augmented_train_validation"
start = 0
window = 60
windowDistance = 20
count = 7


config = {}
config["epochs"] = 500
config["batch_size"] = 64
config["verbose"] = 1

layer1 = {
	"num_filters":10,
	"kernel_size":5,
	"dropout":.5,
	"pool_size":2
}

layer2 = {
	"num_filters":20,
	"kernel_size":3,
	"dropout":.5,
	"pool_size":4
}

config["conv_layers"] = [layer2]
config["dense_layers"] = [10]

with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)

models = []
predictionsList = []

startWindows = []

trainFeatures, trainLabels = Helper.getTrainableData(trainSet, start, window)
cnn = CNN_Re(trainFeatures, trainLabels, modelOption=config)

for i in range(count):
	start = windowDistance * i

	startWindows.append([start, window])

	trainFeatures, trainLabels = Helper.getTrainableData(trainSet, start, window)
	validationFeatures, validationLabels = Helper.getTrainableData(validationSet, start, window)
	testFeatures, testLabels = Helper.getTrainableData(testSet, start, window)

	# fcnn = FCNN(trainFeatures, trainLabels, validationFeatures, validationLabels, config=config)
	model = cnn.trainModel(trainFeatures, trainLabels, validationFeatures, validationLabels)
	# models.append(model)
	# evalu, accuracy = model.evaluate(testFeatures, testLabels, verbose=0)
	# print("----------------------")
	# print("accuracy",accuracy)
	print("done for ", start, window)
	
	
model = cnn.model
for i in range(count):
	start = windowDistance * i
	testFeatures, testLabels = Helper.getTrainableData(testSet, start, window)
	prediction = model.predict(testFeatures)
	predictedLabels = tf.argmax(prediction,axis=1)
	# print(predictedLabels.numpy())
	predictionsList.append(predictedLabels.numpy())

predictions = []
for i in range(len(predictionsList[0])):
	trial = []
	for pred in predictionsList:
		trial.append(pred[i])
	predictions.append(trial)

print(predictions)
predictedLabelsTemp = [Helper.getLabel(Counter(x).most_common(1)[0][0],3) for x in predictions]
rf,realLabelsTemp = Helper.getTrainableDataFCNN(testSet, 0, 150)

total = len(realLabelsTemp)
correct = 0
for i in range(total):
	if np.array_equal(predictedLabelsTemp[i], realLabelsTemp[i]):
		correct = correct + 1
accuracy = float(correct/total)

predictedLabels = tf.argmax(predictedLabelsTemp,axis=1)
realLabels = tf.argmax(realLabelsTemp,axis=1)
confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
print(confusionMatrix)
# confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
# print(confusionMatrix)

homeFolder = Helper.createNewFolderNamed("results/" + experiment, folder)
rootFolder = Helper.createNewFolder(homeFolder)
# modelFolder = Helper.createNewFolderNamed(rootFolder, "model")
confusionMatrixFile = rootFolder + "/confusionMatrix.png"
confusionMatrixNormFile = rootFolder + "/confusionMatrixNormalized.png"
# fileName = modelFolder + "/model.h5"

Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],title="Confusion Matrix : Sliding Window", fileName=confusionMatrixFile)
Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],title="Normalized Confusion Matrix : Sliding Window", fileName=confusionMatrixNormFile, normalize=True)

# # serialize model to JSON
# model_json = model.to_json()
# with open(modelFolder + "/model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(fileName)
# print("Saved model to disk")


output = {}
output["input_folder"] = folder
output["model_config"] = config
output["start_window"] = startWindows
output["train_length"] = len(trainSet)
output["validation_length"] = len(validationSet)
output["test_length"] = len(testSet)
output["accuracy"] = str(accuracy)

with open(rootFolder+"/report.json", "w") as outFile:
	json.dump(output, outFile)



