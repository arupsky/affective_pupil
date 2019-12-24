import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Flatten
from tensorflow.keras import initializers
from Helper import Helper
import json
import tensorflow as tf
# fix random seed for reproducibility
numpy.random.seed(8)

experiment = "LSTM"
# folder = "augmented_5000"
# folder = "augmented"
folder = "raw"
# folder = "augmented_train_validation"
# folder = "rawtest2"
start = 0
window = 240
epochs = 300
batch_size = 32
verbose = 1


with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)

for val in validationSet:
	trainSet.append(val)

print(len(validationSet))

trainFeatures, trainLabels = Helper.getTrainableData(trainSet, start, window)
validationFeatures, validationLabels = Helper.getTrainableData(validationSet, start, window)
testFeatures, testLabels = Helper.getTrainableData(testSet, start, window)

# trainFeatures2, trainLabels2 = Helper.getTrainableData(trainValidation, start, window);

print(trainFeatures.shape)

model = Sequential()
model.add(LSTM(20, input_shape=(trainFeatures.shape[1], trainFeatures.shape[2]), return_sequences=True))
# model.add(LSTM(20, return_sequences=True))
# model.add(TimeDistributed(Dense(20)))
model.add(Flatten())
# model.add(Dense(3, activation="softmax"))
model.add(Dense(trainLabels.shape[1], activation='softmax'))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainFeatures, trainLabels, validation_split=.3, epochs=epochs, batch_size=batch_size, verbose=verbose)

print(model.summary())

# _, accuracy = model.evaluate(testFeatures, testLabels, verbose=0)
accuracy = 0
print("----------------------")
print("accuracy",accuracy)
prediction = model.predict(testFeatures)
predictedLabels = tf.argmax(prediction,axis=1)
realLabels = tf.argmax(testLabels,axis=1)
confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
print(confusionMatrix)
print(model.summary())


homeFolder = Helper.createNewFolderNamed("results/" + experiment, folder)
rootFolder = Helper.createNewFolder(homeFolder)
# modelFolder = Helper.createNewFolderNamed(rootFolder, "model")
confusionMatrixFile = rootFolder + "/confusionMatrix.png"
confusionMatrixNormFile = rootFolder + "/confusionMatrixNormalized.png"
# fileName = modelFolder + "/model.h5"

Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],start=start, window = window, fileName=confusionMatrixFile)
Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],start=start, window = window, fileName=confusionMatrixNormFile, normalize=True)

# # serialize model to JSON
# model_json = model.to_json()
# with open(modelFolder + "/model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(fileName)
# print("Saved model to disk")

output = {}
output["input_folder"] = folder
output["start"] = start
output["window"] = window
# output["model_summary"] = modelOptions
output["train_length"] = len(trainSet)
output["validation_length"] = len(validationSet)
output["test_length"] = len(testSet)
output["accuracy"] = str(accuracy)

with open(rootFolder+"/report.json", "w") as outFile:
	json.dump(output, outFile)


