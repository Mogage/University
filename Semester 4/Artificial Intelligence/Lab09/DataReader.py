import numpy as np
import cv2


class DataReader:
    @staticmethod
    def __shuffleData(inputs, outputs):
        noData = len(inputs)
        permutation = np.random.permutation(noData)
        inputs = inputs[permutation]
        outputs = outputs[permutation]

        return inputs, outputs

    @staticmethod
    def loadIrisData():
        from sklearn.datasets import load_iris

        data = load_iris()
        inputs = data['data']
        outputs = data['target']
        outputNames = data['target_names']
        featureNames = list(data['feature_names'])
        inputs = [
            [feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')],
             feat[featureNames.index('sepal width (cm)')], feat[featureNames.index('petal width (cm)')]]
            for feat in inputs
        ]

        # inputs, outputs = DataReader.__shuffleData(inputs, outputs)

        return inputs, outputs, outputNames

    @staticmethod
    def loadDigitsData():
        from sklearn.datasets import load_digits

        data = load_digits()
        inputs = data.images
        outputs = data['target']
        outputNames = data['target_names']

        inputs, outputs = DataReader.__shuffleData(inputs, outputs)

        return inputs, outputs, outputNames

    @staticmethod
    def loadImagesData(path):
        import os
        labels = [1 if 'sepia' in filename else 0 for filename in os.listdir(path)]
        images = []
        for filename in os.listdir(path):
            image = cv2.imread(os.path.join(path, filename))
            resized = cv2.resize(image, (100, 100))
            images.append(resized)
        return images, labels, ['normal', 'sepia']

    @staticmethod
    def splitData(inputs, outputs):
        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        testSample = [i for i in indexes if i not in trainSample]

        trainInputs = [inputs[i] for i in trainSample]
        trainOutputs = [outputs[i] for i in trainSample]
        testInputs = [inputs[i] for i in testSample]
        testOutputs = [outputs[i] for i in testSample]

        return trainInputs, trainOutputs, testInputs, testOutputs

