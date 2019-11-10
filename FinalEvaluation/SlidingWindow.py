from Helper import Helper
from CNN import CNN
from FCNN import FCNN
import json
import tensorflow as tf
from statistics import mode 
from collections import Counter

# folder = "augmented_5000"
# folder = "augmented"
folder = "raw"
# folder = "augmented_train_validation"
start = 0
window = 30
count = 6

with open("data/splitted/"+folder+"/train.json") as inputFile:
	trainSet = json.load(inputFile)

with open("data/splitted/"+folder+"/validation.json") as inputFile:
	validationSet = json.load(inputFile)

with open("data/splitted/"+folder+"/test.json") as inputFile:
	testSet = json.load(inputFile)

models = []
predictionsList = []

for i in range(count):
	start = window * i

	trainFeatures, trainLabels = Helper.getTrainableDataFCNN(trainSet, start, window)
	validationFeatures, validationLabels = Helper.getTrainableDataFCNN(validationSet, start, window)
	testFeatures, testLabels = Helper.getTrainableDataFCNN(testSet, start, window)

	fcnn = FCNN(trainFeatures, trainLabels, validationFeatures, validationLabels, epochs=50, verbose=0)
	model = fcnn.trainModel()
	models.append(model)
	evalu, accuracy = model.evaluate(testFeatures, testLabels, verbose=0)
	print("----------------------")
	print("accuracy",accuracy)
	print("done for ", start, window)
	
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
predictedLabels = [Counter(x).most_common(1)[0][0] for x in predictions]
print(predictedLabels)
rf,realLabels = Helper.getTrainableDataFCNN(testSet, 0, 150)
confusionMatrix = tf.math.confusion_matrix(realLabels, predictedLabels, 3).numpy()
print(confusionMatrix)




