from SlicedDataFormatter import SlicedDataFormatter
from Helper import Helper
from CNN import CNN
import json
import tensorflow as tf

folder = "augmented_5000"
# folder = "augmented"
# folder = "raw"
# folder = "augmented_train_validation"
start = 0
window = 150

modelOptions = {}
modelOptions["epochs"] = 300
modelOptions["batch_size"] = 64
modelOptions["verbose"] = 1

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

modelOptions["conv_layers"] = [layer1,layer2]
modelOptions["dense_layers"] = [10]


# modelOptions["layer1_num_filters"] = 10
# modelOptions["layer1_kernel_size"] = 5
# modelOptions["layer1_dropout"] = .5
# modelOptions["layer1_pool_size"] = 2

# modelOptions["layer2_num_filters"] = 20
# modelOptions["layer2_kernel_size"] = 3
# modelOptions["layer2_dropout"] = .5
# modelOptions["layer2_pool_size"] = 4

with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)

trainFeatures, trainLabels = Helper.getTrainableData(trainSet, start, window)
validationFeatures, validationLabels = Helper.getTrainableData(validationSet, start, window)
testFeatures, testLabels = Helper.getTrainableData(testSet, start, window)

# cnn = CNN(trainFeatures, trainLabels, validationFeatures, validationLabels, modelOptions)
cnn = CNN(trainFeatures, trainLabels, validationFeatures, validationLabels, epochs=500, verbose=0)
model = cnn.trainModel()
_, accuracy = model.evaluate(testFeatures, testLabels, verbose=0)
print("----------------------")
print("accuracy",accuracy)
prediction = model.predict(testFeatures)
predictedLabels = tf.argmax(prediction,axis=1)
realLabels = tf.argmax(testLabels,axis=1)
confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
print(confusionMatrix)
print(model.summary())


rootFolder = Helper.createNewFolder("report")
modelFolder = Helper.createNewFolderNamed(rootFolder, "model")
confusionMatrixFile = rootFolder + "/confusionMatrix.png"
confusionMatrixNormFile = rootFolder + "/confusionMatrixNormalized.png"
fileName = modelFolder + "/model.h5"
# tf.keras.models.save_model(model, fileName)
Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],start=start, window = window, fileName=confusionMatrixFile)
Helper.plot_confusion_matrix(confusionMatrix,["Negative", "Neutral", "positive"],start=start, window = window, fileName=confusionMatrixNormFile, normalize=True)

# serialize model to JSON
model_json = model.to_json()
with open(modelFolder + "/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(fileName)
print("Saved model to disk")

output = {}
output["input_folder"] = folder
output["start"] = start
output["window"] = window
output["model_summary"] = modelOptions
output["train_length"] = len(trainSet)
output["validation_length"] = len(validationSet)
output["test_length"] = len(testSet)
output["accuracy"] = str(accuracy)

with open(rootFolder+"/report.json", "w") as outFile:
	json.dump(output, outFile)


