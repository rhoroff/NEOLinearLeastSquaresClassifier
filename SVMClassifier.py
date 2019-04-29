import numpy as np
from sklearn.svm import SVC
import collections, functools, operator 
from LLSClassifier import parse_database_into_matrix, split_data_into_training_and_testing

def train_classifier(classifier, X, y):
    # train the classifier with the data
    classifier.fit(X, y)
    return classifier

def test_classifier(classifier, X):
    # test with one or more data points
    return classifier.predict(X)

def classification_percent_error(errors_per_label, labels):

    # Find the number of classifcations per class
    labels_uniques = np.unique(labels, return_counts=True, axis = 0)

    # Build a dictionary of key = label and value = count
    N_per_label = dict((str(key), value) for (key, value) in zip(labels_uniques[0], labels_uniques[1]))

    return {k: errors_per_label[k]/N_per_label[k] for k in N_per_label.keys() & errors_per_label}

def vector_to_classification(vector):
    if vector == [1, 0, 0]:
        return '1'
    elif vector == [0, 1, 0]:
        return '2'
    elif vector == [0, 0, 1]:
        return '3'

def classification_error(labels, predictions):

    # Generate a list of indices for the entire set of labels
    label_indices = list(range(0,labels.shape[0]))

    # Find the indices that are correct
    non_error_indices = np.where(labels == predictions)[0].tolist()

    # Find the indices that are NOT correct
    error_indices = np.setdiff1d(label_indices, non_error_indices)

    # Find the original labels that were misclassified
    error_labels = labels[error_indices]

    # Find the number of errors per class
    error_labels_uniques = np.unique(error_labels, return_counts=True, axis = 0)

    # Build a dictionary of key = label and value = count of errors
    N_errors_per_label = dict((str(key), value) for (key, value) in zip(error_labels_uniques[0], error_labels_uniques[1]))

    return N_errors_per_label

if __name__ == '__main__':

    # data_points, labels = parse_database_into_matrix('/Source/numerical-analysis/NEOLinearLeastSquaresClassifier/WineDatabase/wine.data')
    data_points, labels = parse_database_into_matrix('/Source/numerical-analysis/NEOLinearLeastSquaresClassifier/IrisDatabase/iris.data')

    data_split = split_data_into_training_and_testing([data_points, labels], 50)

    # Training data X -> features, Y -> labels
    training_data = data_split[0][0].T
    training_labels = data_split[0][1]

    training_labels = np.asarray(list(vector_to_classification(v) for v in training_labels.T.tolist()))

    # Testing data X -> features, Y -> labels
    testing_data = data_split[1][0].T
    testing_labels = data_split[1][1]

    # print(testing_data)

    testing_labels = np.asarray(list(vector_to_classification(v) for v in testing_labels.T.tolist()))

    # print(testing_labels)
    # initialize the support vector class
    classifier = SVC(gamma='auto')

    classifier = train_classifier(classifier, training_data, training_labels)

    predictions = test_classifier(classifier, testing_data)

    # Find the misclassification error per class
    print(classification_error(testing_labels, predictions))