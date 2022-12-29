from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from UtilityFunctions import extractDataandLabel
import requests

#this is unused URL
ClassURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"
DataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Get File and make a list of it
Response = requests.get(DataURL)
Response = Response.text.splitlines()
Response.pop()

# Get Data and Labels from Response
Data, Label = extractDataandLabel(Response)

#Split data to train data and test data
TrainData, TestData, TrainLabel, TestLabel = train_test_split(Data,Label,train_size=0.8,test_size=0.2,random_state=12)

#Train Model
knn = KNeighborsClassifier()
knn.fit(TrainData,TrainLabel)

#Make Predictions
Predictions = knn.predict(TestData)
print(accuracy_score(Predictions, TestLabel))