import random
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class ANN:
    @staticmethod
    def __sigmoid(number):
        return 1 / (1 + np.exp(-number))

    @staticmethod
    def __derivativeSigmoid(number):
        sig = ANN.__sigmoid(number)
        return sig * (1 - sig)

    def __init__(self, noHiddenLayers=25, activationFunction='sigmoid', learningRate=0.0001,
                 noEpochs=300):
        self.noHiddenLayers = noHiddenLayers
        self.activationFunction = getattr(self, '_ANN__' + activationFunction)
        self.derivativeFunction = getattr(self,
                                          '_ANN__derivative' + activationFunction[0].upper() + activationFunction[1:])
        self.learningRate = learningRate
        self.noEpochs = noEpochs
        self.network = []

    def __initNetwork(self, noInputs, noOutputs, noHiddenLayers):
        self.network.append(np.random.randn(noInputs, noHiddenLayers))
        self.network.append(np.random.randn(noHiddenLayers, noOutputs))

    def __forward(self, inputs):
        self.hiddenActivation = self.activationFunction(
            np.dot(inputs, self.network[0]))
        outputActivation = self.activationFunction(
            np.dot(self.hiddenActivation, self.network[1]))
        return outputActivation

    def __backward(self, inputs, expected, output):
        outputError = expected - output
        outputDelta = np.dot(self.hiddenActivation.T, outputError)
        hiddenDelta = np.dot(inputs.T, np.dot(outputError, self.network[1].T) * self.hiddenActivation * (
                    1 - self.hiddenActivation))
        self.network[0] += self.learningRate * hiddenDelta * self.derivativeFunction(self.network[0])
        self.network[1] += self.learningRate * outputDelta * self.derivativeFunction(self.network[1])

    def fit(self, inputs, outputs):
        lb = LabelBinarizer()
        noOutputs = len(set(outputs))
        outputs = lb.fit_transform(outputs)

        fitInputs = np.array(inputs)

        self.__initNetwork(len(inputs[0]), noOutputs, self.noHiddenLayers)

        for _ in range(self.noEpochs):
            output = self.__forward(fitInputs)
            self.__backward(inputs, outputs, output)

    def predict(self, inputs):
        return np.argmax(self.__forward(inputs), axis=1)
