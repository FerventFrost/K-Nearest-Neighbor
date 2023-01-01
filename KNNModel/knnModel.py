from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as Metrics 
import UtilityFunctions as Utility
import requests

#this is unused URL
ClassURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"
DataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Get File and make a list of it
Response = requests.get(DataURL)
Response = Response.text.splitlines()
Response.pop()

# Get Data and Labels from Response
Data, Label = Utility.extractDataandLabel(Response)

#Split data to train data and test data
TrainData, TestData, TrainLabel, TestLabel = train_test_split(Data,Label,train_size=0.8,test_size=0.2,random_state=12)

#Train Model
knn = KNeighborsClassifier()
knn.fit(TrainData,TrainLabel)

#Make Predictions
Predictions = knn.predict(TestData)
Accuracy = Metrics.accuracy_score(TestLabel, Predictions)
Recall = Metrics.recall_score(TestLabel, Predictions, average='micro')

#Print Results
print(f"Model Accuracy is {Accuracy}")
print(F"Model Error Rate is {1 - Accuracy}")
print(f"Model Sensitivity is {Recall}")

#Confusion Matrix
matrix_confusion = Metrics.confusion_matrix(TestLabel, Predictions, labels=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])
IrisSetosa, IrisVirginica, IrisVersicolor = Utility.IrisClassifierTable(matrix_confusion)

#Model Accuracy
SetosaAccuracy = (IrisSetosa[0] + IrisSetosa[3]) / sum(IrisSetosa)
VirginicaAccuracy = (IrisVirginica[0] + IrisVirginica[3]) / sum(IrisVirginica)
VersicolorAccuracy = (IrisVersicolor[0] + IrisVersicolor[3]) / sum(IrisVersicolor)
TotalAccuracy = (SetosaAccuracy + VirginicaAccuracy + VersicolorAccuracy) / 3
print(f"Model Accuracy using confusion matirx is {TotalAccuracy}")

#Model Error Rate
print(f"Model Error Rate using confusion matirx is {1- TotalAccuracy}")

#Model False Positive Rate
SetosaFPR = IrisSetosa[1] / (IrisSetosa[1] + IrisSetosa[3])
VirginicaFPR = IrisVirginica[1] / (IrisVirginica[1] + IrisVirginica[3])
VersicolorFPR = IrisVersicolor[1] / (IrisVersicolor[1] + IrisVersicolor[3])
TotalFPR = (SetosaFPR + VirginicaFPR + VersicolorFPR) / 3
print(f"Model False Positive Rate using confusion matirx is {TotalFPR}")

#Model Recall
SetosaRecall = IrisSetosa[0] / (IrisSetosa[0] +  IrisSetosa[2])
VirginicaRecall = IrisVirginica[0] / (IrisVirginica[0] + IrisVirginica[2])
VersicolorRecall = IrisVersicolor[0] / (IrisVersicolor[0] + IrisVersicolor[2])
TotalRecall = (SetosaRecall + VirginicaRecall + VersicolorRecall) / 3
print(f"Model Recall using confusion matirx is {TotalRecall}")

#Model Precision
SetosaPrecision = IrisSetosa[0] / (IrisSetosa[0] +  IrisSetosa[1])
VirginicaPrecision = IrisVirginica[0] / (IrisVirginica[0] + IrisVirginica[1])
VersicolorPrecision = IrisVersicolor[0] / (IrisVersicolor[0] + IrisVersicolor[1])
TotalPrecision = (SetosaPrecision + VirginicaPrecision + VersicolorPrecision) / 3
print(f"Model Precision using confusion matirx is {TotalPrecision}")

#Model F-Measure
f1_score = (2 * TotalRecall * TotalPrecision) / (TotalRecall + TotalPrecision)
print(f"Model F-Measure using confusion matirx is {f1_score}")