import random
import matplotlib.pyplot as plt
from DataReader import DataReader
from ANN import ANN
from Normalization import Normalization
import numpy as np
from sklearn import neural_network


class Solver:
    def __init__(self):
        self.__classifier = None

    @staticmethod
    def flatten(mat):
        x = []
        for line in mat:
            for el in line:
                x.append(el)
        return x

    @staticmethod
    def flattenImage(image):
        return image.flatten()

    def __run(self, trainInputs, trainOutputs, testInputs, testOutputs, outputNames):
        self.training(trainInputs, trainOutputs)
        predictedLabels = self.classification(testInputs)
        acc, prec, recall, cm = self.evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)
        print('acc: ', acc)
        print('precision: ', prec)
        print('recall: ', recall)

        return predictedLabels

    def __showImages(self, testInputs, testOutputs, predictedLabels):
        n = 10
        m = 5
        fig, axes = plt.subplots(n, m, figsize=(7, 7))
        fig.tight_layout()
        for i in range(0, n):
            for j in range(0, m):
                axes[i][j].imshow(testInputs[m * i + j])
                if testOutputs[m * i + j] == predictedLabels[m * i + j]:
                    font = 'normal'
                else:
                    font = 'bold'
                axes[i][j].set_title(
                    'real ' + str(testOutputs[m * i + j]) + '\npredicted ' + str(predictedLabels[m * i + j]),
                    fontweight=font)
                axes[i][j].set_axis_off()

        plt.show()

    def training(self, trainInputs, trainOutputs):
        self.__classifier.fit(trainInputs, trainOutputs)

    def classification(self, testInputs):
        return self.__classifier.predict(testInputs)

    def evalMultiClass(self, realLabels, computedLabels, labelNames):
        from sklearn.metrics import confusion_matrix

        confMatrix = confusion_matrix(realLabels, computedLabels)
        acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
        precision = {}
        recall = {}
        for i in range(len(labelNames)):
            precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
            recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
        return acc, precision, recall, confMatrix

    def solveIris(self, custom=True):
        inputs, outputs, outputNames = DataReader.loadIrisData()
        trainInputs, trainOutputs, testInputs, testOutputs = DataReader.splitData(inputs, outputs)
        normalize = Normalization()
        normalize.fit(trainInputs)
        trainInputs = np.array(normalize.transform(trainInputs))
        testInputs = np.array(normalize.transform(testInputs))

        random.seed(1)
        if custom:
            self.__classifier = ANN()
        else:
            self.__classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=100,
                                                             solver='sgd', verbose=100, random_state=1,
                                                             learning_rate_init=.1)

        self.__run(trainInputs, trainOutputs, testInputs, testOutputs, outputNames)

    def solveDigits(self, custom=True):
        inputs, outputs, outputNames = DataReader.loadDigitsData()
        trainInputs, trainOutputs, testInputs, testOutputs = DataReader.splitData(inputs, outputs)

        normalize = Normalization()
        flattenTrainInputs = [self.flatten(element) for element in trainInputs]
        flattenTestInputs = [self.flatten(element) for element in testInputs]
        normalize.fit(flattenTrainInputs)
        trainInputsNormalised = np.array(normalize.transform(flattenTrainInputs))
        testInputsNormalised = np.array(normalize.transform(flattenTestInputs))

        random.seed(1)
        if custom:
            self.__classifier = ANN()
        else:
            self.__classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100,
                                                             solver='sgd', verbose=10, random_state=1,
                                                             learning_rate_init=.1)

        predictedLabels = self.__run(trainInputsNormalised, trainOutputs, testInputsNormalised, testOutputs,
                                     outputNames)
        self.__showImages(testInputs, testOutputs, predictedLabels)

    def solveImagesAnn(self):
        from sklearn.model_selection import train_test_split
        inputs, outputs, outputNames = DataReader.loadImagesData('pictures')
        trainInputs, testInputs, trainOutputs, testOutputs = train_test_split(inputs, outputs, test_size=0.5, random_state=1)

        normalize = Normalization()
        flattenTrainInputs = [self.flattenImage(element) for element in trainInputs]
        flattenTestInputs = [self.flattenImage(element) for element in testInputs]
        normalize.fit(flattenTrainInputs)
        trainInputsNormalised = np.array(normalize.transform(flattenTrainInputs))
        testInputsNormalised = np.array(normalize.transform(flattenTestInputs))

        random.seed(1)
        self.__classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', alpha=0.0001,
                                                         max_iter=500, solver='sgd', verbose=True, random_state=1,
                                                         learning_rate_init=.001)

        self.__run(trainInputsNormalised, trainOutputs, testInputsNormalised, testOutputs,
                                     outputNames)
        
    def solveImagesCnn(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import os
        import cv2

        dataDir = 'pictures'
        imageSize = (30, 30)

        def resize_image(imagePath):
            image = cv2.imread(imagePath)
            image = cv2.resize(image, imageSize)
            return image

        data = []
        labels = []
        testData = []
        testLabels = []
        totalSamples = 0
        for imagePath in os.listdir(dataDir):
            image = resize_image(os.path.join(dataDir, imagePath))
            # test if image filename contains sepia
            if 'test' in imagePath:
                testData.append(image)
                if 'sepia' in imagePath:
                    testLabels.append(1)
                else:
                    testLabels.append(0)

            else:
                totalSamples += 1
                data.append(image)
                if 'sepia' in imagePath:
                    labels.append(1)
                else:
                    labels.append(0)

        testData = np.array(testData)
        testLabels = np.array(testLabels)

        data = np.array(data)
        labels = np.array(labels)

        trainIndices = np.random.choice(totalSamples, int(totalSamples * 0.8), replace=False)
        testIndices = np.array(list(set(range(totalSamples)) - set(trainIndices)))

        trainInput = data[trainIndices]
        trainOutput = labels[trainIndices]
        testInput = data[testIndices]
        testOutput = labels[testIndices]

        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(trainInput, trainOutput, epochs=100, batch_size=32, validation_data=(testInput, testOutput))

        loss, acc = model.evaluate(testData, testLabels, verbose=2)

        print('Test accuracy:', acc)

