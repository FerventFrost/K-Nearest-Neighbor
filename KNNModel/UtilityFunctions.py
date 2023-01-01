from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as Metrics

# Extract Data From the Iris Dataset Response and Convert it into Arrays
def extractDataandLabel(Response) -> tuple:
    Data = []
    Label = []
    for Values in Response:
        D1, D2, D3, D4, D5 = Values.split(",")
        D = list(map(float, [D1,D2,D3,D4] ))
        Data.append( D )
        Label.append(D5)

    return (Data, Label)

# Construct Classifier table for every Iris
def IrisClassifierTable(ConfusionMatrix):
    IrisSetosa = []
    IrisVirginica = []
    IrisVersicolor = []

    # Iris Setosa Classifer Table
    IrisSetosa.append( ConfusionMatrix[0][0] )
    IrisSetosa.append( (ConfusionMatrix[0][1] + ConfusionMatrix[0][2]) )
    IrisSetosa.append( (ConfusionMatrix[1][0] + ConfusionMatrix[2][0]) )
    IrisSetosa.append( (ConfusionMatrix[1][1] + ConfusionMatrix[1][2] + ConfusionMatrix[2][1] + ConfusionMatrix[2][2]) )

    # Iris Virginica Classifer Table
    IrisVirginica.append( ConfusionMatrix[1][1] )
    IrisVirginica.append( (ConfusionMatrix[1][0] + ConfusionMatrix[1][2]) )
    IrisVirginica.append( (ConfusionMatrix[0][1] + ConfusionMatrix[2][1]) )
    IrisVirginica.append( (ConfusionMatrix[0][0] + ConfusionMatrix[0][2] + ConfusionMatrix[2][0] + ConfusionMatrix[2][2]) )

    # Iris Versicolor Classifer Table
    IrisVersicolor.append( ConfusionMatrix[2][2] )
    IrisVersicolor.append( (ConfusionMatrix[2][0] + ConfusionMatrix[2][1]) )
    IrisVersicolor.append( (ConfusionMatrix[0][2] + ConfusionMatrix[1][2]) )
    IrisVersicolor.append( (ConfusionMatrix[0][0] + ConfusionMatrix[0][1] + ConfusionMatrix[1][0] + ConfusionMatrix[1][1]) )

    return (IrisSetosa, IrisVirginica, IrisVersicolor)

# misclassification rate
def ErrorRate(Data, Label, Prediction):
    TotalPrediction = len(Data)
    IncorrectPrediction = TotalPrediction - Metrics.accuracy_score(Prediction, Label, normalize=False)
    return IncorrectPrediction / TotalPrediction