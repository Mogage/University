import random
from math import exp
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoidDerivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


class Regression:
    def __init__(self):
        self.__intercept = 0.0
        self.__coefficients = []

    @property
    def intercept(self):
        return self.__intercept

    @property
    def coefficients(self):
        return self.__coefficients

    def fit(self, independent, dependent, learningRate=0.1, noEpochs=100):
        self.__coefficients = [0.0 for _ in range(len(independent[0]) + 1)]

        for epoch in range(noEpochs):
            errors = [0] * len(independent[0])

            for i in range(len(independent)):
                computed = sigmoid(self.eval(independent[i], self.__coefficients))
                crtError = computed - dependent[i]
                derivative = sigmoidDerivative(self.eval(independent[i], self.__coefficients))

                for j in range(len(independent[0])):
                    errors[j] += crtError * derivative * independent[i][j]

            for j in range(len(independent[0])):
                self.__coefficients[j + 1] = self.__coefficients[j + 1] - learningRate * errors[j]
            self.__coefficients[0] = self.__coefficients[0] - learningRate * sum(errors)

        # for epoch in range(noEpochs):
        #     for i in range(len(independent)):
        #         computed = sigmoid(self.eval(independent[i], self.__coefficients))
        #         crtError = computed - dependent[i]
        #         derivative = sigmoidDerivative(self.eval(independent[i], self.__coefficients))
        #
        #         for j in range(0, len(independent[0])):
        #             self.__coefficients[j + 1] = self.__coefficients[j + 1] - learningRate * crtError * derivative * \
        #                                          independent[i][j]
        #         self.__coefficients[0] = self.__coefficients[0] - learningRate * crtError * derivative * 1

        self.__intercept = self.__coefficients[0]
        self.__coefficients = self.__coefficients[1:]

    def eval(self, feature, coefficients):
        dependent = coefficients[0]
        for j in range(len(feature)):
            dependent += coefficients[j + 1] * feature[j]
        return dependent

    def predictOneSample(self, features):
        threshold = 0.5
        yi = sigmoid(self.eval(features, [self.__intercept] + [c for c in self.__coefficients]))
        return 0 if yi < threshold else 1

    def predict(self, inputs):
        computed = [self.predictOneSample(sample) for sample in inputs]
        return computed


class MyLogisticRegression:
    def __init__(self):
        self.__intercept = None
        self.__coefficients = None
        self.__classes = None

    def fit(self, independent, dependent):
        self.__intercept = []
        self.__coefficients = []
        self.__classes = list(set(dependent))

        for classLabel in self.__classes:
            yi = [int(val == classLabel) for val in dependent]
            classifier = Regression()
            classifier.fit(independent, yi)

            self.__intercept.append(classifier.intercept)
            self.__coefficients.append(classifier.coefficients)

    def eval(self, feature, coefficients):
        dependent = coefficients[0]
        for j in range(len(feature)):
            dependent += coefficients[j + 1] * feature[j]
        return dependent

    def predict(self, inTest):
        predictions = []
        for sample in inTest:
            predictionsForClasses = []
            for i in range(len(self.__classes)):
                predictionsForClasses.append(sigmoid(self.eval(sample, [self.__intercept[i]] + self.__coefficients[i])))
            predictions.append(self.__classes[np.argmax(predictionsForClasses)])
        return predictions
