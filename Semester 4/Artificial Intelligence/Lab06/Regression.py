class Regression:
    def __init__(self):
        self.__w = [0.0, 0.0, 0.0]

    @property
    def w(self):
        return self.__w

    def fit(self, independent, dependent1, dependent2):
        length = len(dependent1)
        independentSum = sum(independent)
        independentI = [independent[i] - independentSum / length for i in range(length)]
        dependent1Sum = sum(dependent1)
        dependent1I = [dependent1[i] - dependent1Sum / length for i in range(length)]
        dependent2Sum = sum(dependent2)
        dependent2I = [dependent2[i] - dependent2Sum / length for i in range(length)]

        dependent1SquaredSum = sum([i * i for i in dependent1I])
        dependent2SquaredSum = sum([i * i for i in dependent2I])
        dependent1Dependent2Sum = sum([i * j for i, j in zip(dependent1I, dependent2I)])
        independentDependent1Sum = sum([i * j for i, j in zip(independentI, dependent1I)])
        independentDependent2Sum = sum([i * j for i, j in zip(independentI, dependent2I)])

        self.__w[1] = (independentDependent1Sum * dependent2SquaredSum - independentDependent2Sum * dependent1Dependent2Sum) / \
                      (dependent1SquaredSum * dependent2SquaredSum - dependent1Dependent2Sum ** 2)
        self.__w[2] = (independentDependent2Sum * dependent1SquaredSum - independentDependent1Sum * dependent1Dependent2Sum) / \
                      (dependent1SquaredSum * dependent2SquaredSum - dependent1Dependent2Sum ** 2)
        self.__w[0] = independentSum / length - self.__w[1] * dependent1Sum / length - self.__w[2] * dependent2Sum / \
                      length

    def predict(self, features):
        return [self.__w[0] + self.__w[1] * x1 + self.__w[2] * x2 for x1, x2 in features]
