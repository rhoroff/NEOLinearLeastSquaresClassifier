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
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [0,0]])
    Y = np.array([1, 1, 2, 2, 3])

    # initialize the support vector class
    classifier = SVC(gamma='auto')

    classifier = train_classifier(classifier, X, Y)

    print(test_classifier(classifier, [[1,0]]))