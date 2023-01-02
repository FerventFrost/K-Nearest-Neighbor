from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as Metrics 
import UtilityFunctions as Utility
import requests

#Constants
TP = 0
FP = 1
FN = 2
TN = 3

#this is unused URL
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
print(f'''
###################################################################
SKLearn Metrics Calculations
###################################################################
''')
print(f"Model Accuracy is {Accuracy}")
print(F"Model Error Rate is {FP - Accuracy}")
print(f"Model Sensitivity is {Recall}")

#Confusion Matrix
print(f'''
###################################################################
Confusion Matrix Calculations
###################################################################
''')
matrix_confusion = Metrics.confusion_matrix(TestLabel, Predictions, labels=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])
IrisSetosa, IrisVirginica, IrisVersicolor = Utility.IrisClassifierTable(matrix_confusion)

#Model Accuracy
SetosaAccuracy = (IrisSetosa[TP] + IrisSetosa[TN]) / sum(IrisSetosa)
VirginicaAccuracy = (IrisVirginica[TP] + IrisVirginica[TN]) / sum(IrisVirginica)
VersicolorAccuracy = (IrisVersicolor[TP] + IrisVersicolor[TN]) / sum(IrisVersicolor)
TotalAccuracy = (SetosaAccuracy + VirginicaAccuracy + VersicolorAccuracy) / 3
print(f"Model Accuracy using confusion matirx is {TotalAccuracy}")

#Model Error Rate
print(f"Model Error Rate using confusion matirx is {FP- TotalAccuracy}")

#Model False Positive Rate
SetosaFPR = IrisSetosa[FP] / (IrisSetosa[FP] + IrisSetosa[TN])
VirginicaFPR = IrisVirginica[FP] / (IrisVirginica[FP] + IrisVirginica[TN])
VersicolorFPR = IrisVersicolor[FP] / (IrisVersicolor[FP] + IrisVersicolor[TN])
TotalFPR = (SetosaFPR + VirginicaFPR + VersicolorFPR) / 3
print(f"Model False Positive Rate using confusion matirx is {TotalFPR}")

#Model Recall
SetosaRecall = IrisSetosa[TP] / (IrisSetosa[TP] +  IrisSetosa[FN])
VirginicaRecall = IrisVirginica[TP] / (IrisVirginica[TP] + IrisVirginica[FN])
VersicolorRecall = IrisVersicolor[TP] / (IrisVersicolor[TP] + IrisVersicolor[FN])
TotalRecall = (SetosaRecall + VirginicaRecall + VersicolorRecall) / 3
print(f"Model Recall using confusion matirx is {TotalRecall}")

#Model Precision
SetosaPrecision = IrisSetosa[TP] / (IrisSetosa[TP] +  IrisSetosa[FP])
VirginicaPrecision = IrisVirginica[TP] / (IrisVirginica[TP] + IrisVirginica[FP])
VersicolorPrecision = IrisVersicolor[TP] / (IrisVersicolor[TP] + IrisVersicolor[FP])
TotalPrecision = (SetosaPrecision + VirginicaPrecision + VersicolorPrecision) / 3
print(f"Model Precision using confusion matirx is {TotalPrecision}")

#Model F-Measure
f1_score = (FN * TotalRecall * TotalPrecision) / (TotalRecall + TotalPrecision)
print(f"Model F-Measure using confusion matirx is {f1_score}")



#Calculate Precision Confusion Matrix
print(f'''
###################################################################
Prescion Calcualtion Using Precision Matrix Directly
###################################################################
''')
IrisP = (matrix_confusion[0][0]) / sum(matrix_confusion[0])
VP = (matrix_confusion[1][1]) / sum(matrix_confusion[1])
MP = (matrix_confusion[2][2]) / sum(matrix_confusion[2])
print(f"Model Precsion recalcu is {(IrisP + VP + MP) / 3}")
