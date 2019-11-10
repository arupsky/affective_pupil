import json
import tensorflow as tf
from keras.models import load_model
from Helper import Helper
from keras.models import model_from_json


start = 0
window = 150
folder = "augmented_5000"
modelFolder = "23"
modelJsonFile = "report/"+modelFolder+"/model/model.json"
modelWeightFile = "report/"+modelFolder+"/model/model.h5"

# load json and create model
json_file = open(modelJsonFile, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(modelWeightFile)
print("Loaded model from disk")

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)

testFeatures, testLabels = Helper.getTrainableData(testSet, start, window)

model = load_model(modelFile)
_, accuracy = model.evaluate(testFeatures, testLabels, verbose=0)
print("----------------------")
print("accuracy",accuracy)
prediction = model.predict(testFeatures)
predictedLabels = tf.argmax(prediction,axis=1)
realLabels = tf.argmax(testLabels,axis=1)
confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
print(confusionMatrix)
print(model.summary())