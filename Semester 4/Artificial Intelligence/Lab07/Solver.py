import numpy as np
from NormalizationFactory import NormalizationFactory as Normalization
from RegressionFactory import Regression


class Solver:
    def __init__(self, inputs, output, multivariate=False):
        self.__inputs = inputs
        self.__output = output
        self.__trainingData = [[], []]
        self.__testingData = [[], []]
        self.__multivariate = multivariate
        self.__regression = Regression()

    def splitTrainTest(self, ratio=0.8, ntype='statistical'):
        np.random.seed(5)
        normalization = Normalization()

        indexes = [i for i in range(len(self.__inputs))]
        trainSample = np.random.choice(indexes, int(ratio * len(self.__inputs)), replace=False)
        testSample = [i for i in indexes if i not in trainSample]

        self.__trainingData[0] = [self.__inputs[i] for i in trainSample]
        self.__trainingData[1] = [self.__output[i] for i in trainSample]
        self.__testingData[0] = [self.__inputs[i] for i in testSample]
        self.__testingData[1] = [self.__output[i] for i in testSample]

        if not self.__multivariate:
            return

        self.__trainingData[0] = normalization.normalize(self.__trainingData[0], ntype)
        self.__trainingData[1] = normalization.normalize(self.__trainingData[1], ntype)
        self.__testingData[0] = normalization.normalize(self.__testingData[0], ntype)
        self.__testingData[1] = normalization.normalize(self.__testingData[1], ntype)

    def train(self):
        self.__regression.fit(self.__trainingData[0], self.__trainingData[1])

    def test(self):
        predictions = self.__regression.predict(self.__testingData[0])
        error = 0.0
        for t1, t2 in zip(predictions, self.__testingData[1]):
            error += (t1 - t2) ** 2
        error = error / len(self.__testingData[1])
        print(error)
