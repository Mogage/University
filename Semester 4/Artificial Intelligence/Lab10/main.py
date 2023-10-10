import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from KMeansCluster import KMeansCluster
import spacy


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
    np.random.seed(5)

    noSamples = len(inputs)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def __representation1_2(trainInputs, testInputs, first=True):
    from sklearn.feature_extraction.text import CountVectorizer

    if first:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(ngram_range=(2, 2))

    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.transform(testInputs)

    return trainFeatures, testFeatures


def __representation3(trainInputs, testInputs):
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(n_features=500)
    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.transform(testInputs)
    return trainFeatures, testFeatures

def __representation4(trainInputs, testInputs):
    from nltk.tokenize import word_tokenize
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    taggedTrainInputs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                         enumerate(trainInputs)]
    taggedTestInputs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                        enumerate(testInputs)]
    max_epochs = 50
    alpha = 0.025

    model = Doc2Vec(
        alpha=alpha,
        min_alpha=0.00025,
        min_count=1,
        dm=1)

    model.build_vocab(taggedTrainInputs)

    for epoch in range(max_epochs):
        model.train(taggedTrainInputs,
                    total_examples=model.corpus_count,
                    epochs=50)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    trainFeatures = [model.infer_vector(doc.words) for doc in taggedTrainInputs]
    testFeatures = [model.infer_vector(doc.words) for doc in taggedTestInputs]
    
    return trainFeatures, testFeatures


def extractCharacteristics(trainInputs, testInputs):
    # return __representation1_2(trainInputs, testInputs)

    # return __representation1_2(trainInputs, testInputs, False)

    # return __representation3(trainInputs, testInputs)

    return __representation4(trainInputs, testInputs)


def shuffleData(inputs, outputs):
    permutation = np.random.permutation(len(inputs))
    inputs = [inputs[el] for el in permutation]
    outputs = [outputs[el] for el in permutation]
    return inputs, outputs


if __name__ == '__main__':
    # iris = load_iris()
    # inputs = iris.data
    # outputs = iris.target
    # inputs, outputs = shuffleData(inputs, outputs)
    # trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    # kmeans = KMeansCluster(n_clusters=3)
    #
    # for _ in range(5):
    #     kmeans.fit(trainInputs)
    #     computedTestOutputs = kmeans.evaluate(testInputs)
    #     print("KMeans code acc: ", accuracy_score(testOutputs, computedTestOutputs))
    #
    # print("---------------------------------------------------------------")

    data = readCsv('reviews_mixed.csv')
    inputs = [data[i][0] for i in range(len(data))]
    outputs = [data[i][1] for i in range(len(data))]
    labelNames = list(set(outputs))

    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    trainFeatures, testFeatures = extractCharacteristics(trainInputs, testInputs)

    from sklearn.cluster import KMeans

    unsupervisedClassifier = KMeans(n_clusters=2, random_state=1, n_init=5)
    unsupervisedClassifier.fit(trainFeatures)

    computedTestIndexes = unsupervisedClassifier.predict(testFeatures)
    computedTestOutputs = [labelNames[value] for value in computedTestIndexes]

    print("Unsupervised acc: ", accuracy_score(testOutputs, computedTestOutputs))

    from sklearn.linear_model import SGDClassifier

    supervisedClassifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=500,
                                         tol=None)
    supervisedClassifier.fit(trainFeatures, trainOutputs)
    computedTestOutputs = supervisedClassifier.predict(testFeatures)
    print("Supervised acc: ", accuracy_score(testOutputs, computedTestOutputs))

    from sklearn import neural_network

    # trainFeatures = trainFeatures.toarray()
    # testFeatures = testFeatures.toarray()

    unsupervisedClassifier = KMeans(n_clusters=100, random_state=0, n_init=5)
    x = unsupervisedClassifier.fit_transform(trainFeatures)
    pos = np.argmin(x, axis=0)
    chosenInputs = [trainFeatures[index] for index in pos]
    chosenOutputs = [list(trainOutputs)[index] for index in pos]
    classifier = neural_network.MLPClassifier(max_iter=500)
    classifier.fit(chosenInputs, chosenOutputs)
    computedTestOutputs = classifier.predict(testFeatures)

    accuracy = accuracy_score(testOutputs, computedTestOutputs)
    print("Semi supervised: ", accuracy)
