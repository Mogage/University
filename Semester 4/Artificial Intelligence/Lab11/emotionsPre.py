import urllib.request
import cv2
import numpy as np
import tensorflow as tf
import glob
from keras.utils import img_to_array


def classify_emotion(image_path, model):
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (48, 48))
    # alternative to cv2.imread
    from PIL import Image

    img = Image.open(image_path).convert('L')
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = img_to_array(img)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=None)[0]
    emotion_indices = [0, 1, 2, 3, 4, 5, 6]
    predicted_label = emotion_indices[prediction.argmax()]
    return predicted_label


def getPredict(path, count, model):
    predictedLabels = []
    image_paths = glob.glob(path)
    shuffle = np.random.permutation(len(image_paths))
    indices = shuffle[:int(len(image_paths) * 0.2)]
    image_paths = [image_paths[i] for i in indices]
    testLabels = [count for _ in range(len(image_paths))]
    for image_path in image_paths:
        predicted_emotion = classify_emotion(image_path, model)
        predictedLabels.append(predicted_emotion)
    print(path.split('/')[1])

    return testLabels, predictedLabels


def run():
    model_url = "https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/emotion_detector_models/model_v6_23.hdf5?raw=true"
    urllib.request.urlretrieve(model_url, "model_v6_23.hdf5")

    model = tf.keras.models.load_model("model_v6_23.hdf5")

    testLabels = []
    predictedLabels = []

    tests, predicts = getPredict('test/angry/*.jpg', 0, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/disgust/*.jpg', 1, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/fear/*.jpg', 2, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/happy/*.jpg', 3, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/neutral/*.jpg', 4, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/sad/*.jpg', 5, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)
    tests, predicts = getPredict('test/surprise/*.jpg', 6, model)
    for test in tests:
        testLabels.append(test)
    for predict in predicts:
        predictedLabels.append(predict)

    accuracy = np.mean(np.equal(testLabels, predictedLabels))
    print('Accuracy: ', accuracy)
