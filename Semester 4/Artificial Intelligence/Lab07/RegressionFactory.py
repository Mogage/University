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

    def __eval(self, feature):
        dependent = feature[-1]
        for index in range(len(feature)):
            dependent += self.__coefficients[index] * feature[index]
        return dependent

    def fit(self, independent, dependent, learningRate=0.001, noOfEpochs=1000):
        self.__coefficients = [0.0 for _ in range(len(independent[0]) + 1)]
        for _ in range(noOfEpochs):
            errors = []

            for index in range(len(independent)):
                computedDependent = self.__eval(independent[index])
                errors.append(computedDependent - dependent[index])

            for index in range(len(independent)):
                for index2 in range(len(independent[0])):
                    self.__coefficients[index2] = self.__coefficients[index2] - learningRate * errors[index] * \
                                                  independent[index][index2]
                self.__coefficients[-1] = self.__coefficients[-1] - learningRate * errors[index]

        self.__intercept = self.__coefficients[-1]
        self.__coefficients = self.__coefficients[:-1]

    def predict(self, features):
        return [self.__eval(feature) for feature in features]
