import numpy as np
from DataReader import DataReader
from Regression import MyLogisticRegression
from Normalisation import CustomNormalisation
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score


def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
    return acc, precision, recall, confMatrix


def runSimple():
    classifier.fit(trainInputs, trainOutputs)
    computedTestOutputs = classifier.predict(testInputs)

    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), computedTestOutputs, outputNames)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)


def runMultiClass():
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    maeScores = cross_val_score(classifier, inputs, outputs, scoring='neg_mean_absolute_error', cv=cv)
    mseScores = cross_val_score(classifier, inputs, outputs, scoring='neg_mean_squared_error', cv=cv)

    print("MAE: %0.3f (+/- %0.3f)" % (-maeScores.mean(), maeScores.std() * 2))
    print("MSE: %0.3f (+/- %0.3f)" % (-mseScores.mean(), mseScores.std() * 2))

 
class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        self.model = MyLogisticRegression()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    inputs, outputs, outputNames = DataReader.loadIrisData()
    trainInputs, trainOutputs, testInputs, testOutputs = DataReader.splitData(inputs, outputs)
    normalisationFactory = CustomNormalisation()
    normalisationFactory.fit(trainInputs)
    trainInputs = normalisationFactory.transform(trainInputs)
    testInputs = normalisationFactory.transform(testInputs)

    # from sklearn import linear_model
    # classifier = linear_model.LogisticRegression() # tool
    # classifier = MyLogisticRegression()
    classifier = CustomLogisticRegression()

    # runSimple()
    runMultiClass()
