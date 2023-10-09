
class DataReader:
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

        return inputs, outputs, outputNames

    @staticmethod
    def splitData(inputs, outputs):
        import numpy as np

        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        testSample = [i for i in indexes if i not in trainSample]

        trainInputs = [inputs[i] for i in trainSample]
        trainOutputs = [outputs[i] for i in trainSample]
        testInputs = [inputs[i] for i in testSample]
        testOutputs = [outputs[i] for i in testSample]

        return trainInputs, trainOutputs, testInputs, testOutputs

