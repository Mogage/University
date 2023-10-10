import glob
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

from classifier import GeneralClassifier


def getTrainData(filePath, train_data, train_labels, label):
    image_paths = glob.glob(filePath)
    shuffle = np.random.permutation(len(image_paths))
    indices = shuffle[:80]
    image_paths = [image_paths[i] for i in indices]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        train_data.append(image)
        train_labels.append(label)
    return train_data, train_labels


def getTestData(filePath, test_data, test_labels, label):
    image_paths = glob.glob(filePath)
    shuffle = np.random.permutation(len(image_paths))
    indices = shuffle[:20]
    image_paths = [image_paths[i] for i in indices]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        test_data.append(image)
        test_labels.append(label)
    return test_data, test_labels


def run():
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    train_data, train_labels = getTrainData('train/angry/*.jpg', train_data, train_labels, 0)
    train_data2, train_labels2 = getTrainData('train/disgust/*.jpg', train_data, train_labels, 3)
    train_data += train_data2
    train_labels += train_labels2
    train_data2, train_labels2 = getTrainData('train/fear/*.jpg', train_data, train_labels, 6)
    train_data += train_data2
    train_labels += train_labels2

    test_data, test_labels = getTestData('test/angry/*.jpg', test_data, test_labels, 0)
    test_data2, test_labels2 = getTestData('test/disgust/*.jpg', test_data, test_labels, 3)
    test_data += test_data2
    test_labels += test_labels2
    test_data2, test_labels2 = getTestData('test/fear/*.jpg', test_data, test_labels, 6)
    test_data += test_data2
    test_labels += test_labels2

    classifier = GeneralClassifier()
    classifier.fit(train_data, train_labels)

    predictions = classifier.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    print("Accuracy for ImageClassifier with manual extraction:", accuracy)

    classifier = GeneralClassifier(True)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    print("Accuracy for ImageClassifier with automatic extraction:", accuracy)
