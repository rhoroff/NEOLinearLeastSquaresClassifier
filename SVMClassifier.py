import numpy as np
from sklearn.svm import SVC

# Training data X -> features, Y -> labels
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])

# initialize the support vector class
classifier = SVC(gamma='auto')

# train the classifier with the data
classifier.fit(X,Y)

# test with one or more data points
print(classifier.predict(X))