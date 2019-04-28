import numpy as np
from sklearn.svm import SVC

def train_classifier(classifier, X, y):
    # train the classifier with the data
    classifier.fit(X, y)
    return classifier

def test_classifier(classifier, X):
    # test with one or more data points
    return classifier.predict(X)

if __name__ == '__main__':
    # Training data X -> features, Y -> labels
    training_data = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [0,0]])
    training_classes = np.array([1, 1, 2, 2, 3])

    # initialize the support vector class
    classifier = SVC(gamma='auto')

    classifier = train_classifier(classifier, training_data, training_classes)

    testing_data = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [0,0], [-9,-9], [-9,-9]])
    testing_classes = np.array([1, 1, 2, 2, 3, 1, 1])
    predictions = test_classifier(classifier, testing_data)
    print(np.where(np.equal(testing_classes, predictions))[0].tolist())
    print(np.setdiff1d(list(range(0,testing_classes.shape[0])), np.where(np.equal(testing_classes, predictions))[0].tolist()))