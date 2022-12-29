
def extractDataandLabel(Response) -> tuple:
    Data = []
    Label = []
    for Values in Response:
        D1, D2, D3, D4, D5 = Values.split(",")
        D = list(map(float, [D1,D2,D3,D4] ))
        Data.append( D )
        Label.append(D5)

    return (Data, Label)