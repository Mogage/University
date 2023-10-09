from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
import cv2
import numpy as np


def readFileNames(fileName):
    with open(fileName) as file:
        lines = file.readlines()
    lines = [line.replace('\n', '') for line in lines]
    lines = [line + '.png' for line in lines]
    fileName = fileName.split('/')
    fileName = fileName[0]
    lines = [fileName + '/' + line for line in lines]
    return lines


def readPhoto(fileName):
    images = []
    for file in fileName:
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resizedImage = cv2.resize(gray, (48, 48))
        preprocessedImage = np.expand_dims(resizedImage, axis=-1)
        images.append(preprocessedImage / 255.0)
    return images


def splitData(happy, sad):
    np.random.seed(5)

    noSamples = len(happy)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [happy[i] for i in trainSample]
    trainOutputs = [0 for _ in trainSample]
    testInputs = [happy[i] for i in testSample]
    testOutputs = [0 for _ in testSample]

    noSamples = len(sad)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs += [sad[i] for i in trainSample]
    trainOutputs += [1 for _ in trainSample]
    testInputs += [sad[i] for i in testSample]
    testOutputs += [1 for _ in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def run():
    happyEmojis = readPhoto(readFileNames("happy_emojis/happy.txt"))
    sadEmojis = readPhoto(readFileNames("sad_emojis/sad.txt"))

    trainInput, trainOutput, testInput, testOutput = splitData(happyEmojis, sadEmojis)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    trainInput = np.asarray(trainInput)
    trainOutput = np.asarray(trainOutput)
    testInput = np.asarray(testInput)
    testOutput = np.asarray(testOutput)

    trainOutput = to_categorical(trainOutput)
    testOutput = to_categorical(testOutput)

    model.fit(trainInput, trainOutput, epochs=100, verbose=2, validation_data=(testInput, testOutput))
    model.evaluate(testInput, testOutput, verbose=2)
