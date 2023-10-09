import csv
import numpy as np
from sklearn.metrics import accuracy_score


def readCsv(fileName):
    data = []
    with open(fileName) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        lineCount = 0
        for row in csvReader:
            if lineCount == 0:
                lineCount = 1
                continue
            else:
                data.append(row)
    return data


def splitData(inputs, outputs):
    noSamples = len(inputs)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def extractCharacteristics(trainInputs, testInputs):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()

    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.transform(testInputs)

    return trainFeatures, testFeatures


def run():
    data = readCsv('reviews_mixed_positive.csv')
    inputs = [data[i][0] for i in range(len(data))]
    outputs = [data[i][1] for i in range(len(data))]

    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    trainFeatures, testFeatures = extractCharacteristics(trainInputs, testInputs)

    from sklearn.linear_model import SGDClassifier

    supervisedClassifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=500,
                                         tol=None)
    supervisedClassifier.fit(trainFeatures, trainOutputs)
    computedTestOutputs = supervisedClassifier.predict(testFeatures)

    values = [1 if output == 'positive' else 0 for output in computedTestOutputs]

    score = np.mean(values)
    print(score)
    if score >= 0.5:
        print("next comment will be positive")
    else:
        print("next comment will be negative")
