import os
import sys
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from keras_facenet import FaceNet


class GeneralClassifier:
    def __init__(self, automate=False):
        if automate:
            self.classifier = ImageClassifier()
        else:
            self.classifier = ImageClassifierFacenet()

    def fit(self, images, labels):
        self.classifier.fit(images, labels)

    def predict(self, images):
        return self.classifier.predict(images)


class ImageClassifier:
    def __init__(self):
        self.classifier = SVC(kernel='linear')

    def extract_hog_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        return hog_features

    def fit(self, images, labels):
        data = []
        for image in images:
            hog_features = self.extract_hog_features(image)
            data.append(hog_features)
        data = np.array(data)
        labels = np.array(labels)
        self.classifier.fit(data, labels)

    def predict(self, images):
        data = []
        for image in images:
            hog_features = self.extract_hog_features(image)
            data.append(hog_features)
        data = np.array(data)
        predictions = self.classifier.predict(data)
        return predictions


class ImageClassifierFacenet:
    def __init__(self):
        self.facenet = FaceNet()
        self.classifier = SVC(verbose=0)

    def preprocess_image(self, image):
        # Preprocess the image for FaceNet model input
        resized_image = cv2.resize(image, (48, 48))
        preprocessed_image = np.expand_dims(resized_image, axis=0)
        return preprocessed_image

    def extract_face_embeddings(self, image):
        preprocessed_image = self.preprocess_image(image)
        # Redirect standard output to a null device
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        embeddings = self.facenet.embeddings(preprocessed_image)[0]

        # Restore standard output
        sys.stdout.close()
        sys.stdout = original_stdout
        return embeddings

    def fit(self, images, labels):
        train_embeddings = []
        train_labels = []
        for image, label in zip(images, labels):
            embeddings = self.extract_face_embeddings(image)
            train_embeddings.append(embeddings)
            train_labels.append(label)
        self.classifier.fit(train_embeddings, train_labels)

    def predict(self, images):
        predictions = []
        for image in images:
            embeddings = self.extract_face_embeddings(image)
            prediction = self.classifier.predict(embeddings.reshape(1, -1))
            predictions.append(prediction)
        return predictions
    