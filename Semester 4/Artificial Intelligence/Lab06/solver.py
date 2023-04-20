import numpy as np

from Regression import Regression


class Solver:
    def __init__(self, dataset, columns):
        self.__data = dataset
        self.__columns = columns
        self.__featureTrain = []
        self.__featureValidation = []
        self.__train = []
        self.__validation = []
        self.__regression = Regression()

    def split_dataset_train_test(self, fraction=0.8):
        np.random.seed(5)
        indexes = [i for i in range(len(self.__data[self.__columns[1]]))]
        trainSample = np.random.choice(indexes, int(fraction * len(self.__data[self.__columns[1]])), replace=False)
        validationSample = [i for i in indexes if i not in trainSample]

        for i in range(1, len(self.__columns)):
            self.__featureTrain.append([self.__data[self.__columns[i]][j] for j in trainSample])
            self.__featureValidation.append([self.__data[self.__columns[i]][j] for j in validationSample])

        # input
        self.__train.append(
            [[self.__featureTrain[0][i], self.__featureTrain[1][i]] for i in range(len(self.__featureTrain[0]))])
        # output
        self.__train.append([self.__data[self.__columns[0]][i] for i in trainSample])

        # input
        self.__validation.append([[self.__featureValidation[0][i], self.__featureValidation[1][i]] for i in
                                  range(len(self.__featureValidation[0]))])
        # output
        self.__validation.append([self.__data[self.__columns[0]][i] for i in validationSample])

    def train(self):
        self.__regression.fit(self.__train[1], self.__featureTrain[0], self.__featureTrain[1])

    def test(self):
        computedValidationOutputs = self.__regression.predict(self.__validation[0])
        error = 0.0
        for t1, t2 in zip(computedValidationOutputs, self.__validation[1]):
            error += (t1 - t2) ** 2
        error = error / len(self.__validation[1])
        w0, w1, w2 = self.__regression.w
        print('the learnt model: f(x) = ' + str(w0) + ' + ' + str(w1) + ' * x1 + ' + str(
            w2) + ' * x2 \nerror = ' + str(error))
        return error
