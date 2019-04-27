import numpy as np
import numpy.linalg as la
import csv, sys, os, re


def classification_to_vector(classification):
    if classification == '1':
        return [1, 0, 0]
    elif classification == '2':
        return [0, 1, 0]
    elif classification == '3':
        return [0, 0, 1]
    elif classification == 'Iris-setosa':
        return [1, 0, 0]
    elif classification == 'Iris-virginica':
        return [0, 1, 0]
    elif classification == 'Iris-versicolor':
        return [0, 0, 1]

def pad_with_one(dataMatrix):
    # Takes in a data matrix and pads it with another dimension of 1's
    data_point_matrix = np.asarray(dataMatrix, dtype=np.float32)
    ones = np.ones((data_point_matrix.shape[0], 1))
    return np.concatenate((data_point_matrix, ones),1)

def parse_database_into_matrix(inputFile):
    # Full class path in case its necessary
    inputFile = os.path.abspath(inputFile)

    # different datasets requires different parsing methods, use regex to filter
    if re.search("iris.data", inputFile):
        with open(inputFile) as csvfile:
            datareader = csv.reader(csvfile)

            # data_point_matrix[0] -> classifications[0]
            data_point_matrix = []
            classifications = []

            for row in datareader:
                data_point = []
                class_col = len(row)-1
                classification = row[class_col]

                # 'Iris-setosa' -> [1,0,0], 'Iris-virginica' -> [0,1,0], 'Iris-versicolor' -> [0,0,1]
                classifications.append(
                    classification_to_vector(classification))
                for i in range(len(row)-1):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)

            #Append a 1 to every data point for the free param
            data_point_matrix = pad_with_one(data_point_matrix)

            return data_point_matrix.T, np.asarray(classifications).T

    elif re.search("wine.data", inputFile):
        with open(inputFile) as csvfile:
            datareader = csv.reader(csvfile)

            # data_point_matrix[0] -> classifications[0]
            data_point_matrix = []
            classifications = []

            for row in datareader:
                data_point = []
                class_col = 0
                classification = row[class_col]

                # '1' -> [1,0,0], '2' -> [0,1,0], '3' -> [0,0,1]
                classifications.append(classification_to_vector(row[0]))

                for i in range(1, len(row)):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)
            #Append a 1 to every data point for the free param
            data_point_matrix = pad_with_one(data_point_matrix)

            return data_point_matrix.T, np.asarray(classifications).T
    else:
        return [], []


def train_weight_vector(inputMatrix, classMatrix, testingLambda):
    """
    Args:
        inputMatrix (2D list): The matrix containing input data points as columns and featrues as rows.
        classMatrix (2D list): The matrix containing classifications for the inputMatrix, maps one to one based on index
        testingLambda (float): The lambda value to be used for conditioning the weight matrix W

    Returns:
        2D numpy.array: The weight matrix W
    """

    X = np.asarray(inputMatrix)
    Y = np.asarray(classMatrix)

    # Arbitrarily small value to start, will vary for testing purposes
    conditioning_lambda = testingLambda
    W = np.dot(la.inv(np.dot(X, X.T) + (conditioning_lambda * np.eye(X.shape[0]))), np.dot(X, Y.T))
    return W

def split_data_into_training_and_testing(database, trainingPercentage):
    """
    Args:
        database (list of np.arrays): A list containing an array of data points as its first entry and an array of classification for the data points as its second entry
                                        The indices of these arrays map one to one
        trainingPercentage (int): The percentage of testing values to split the data points and their classifications into

    Returns:
        list of list of np.arrays: A list containing two lists, the first the training set and its associated labels and the second the testing set and its associated labels
    """
    X = database[0]
    Y = database[1]
    # print(np.unique(Y, axis=1, return_counts=True))
    splitRatio = int(X.shape[1] * (.01*trainingPercentage))
    training = [X[0:,:splitRatio], Y[0:,:splitRatio]]
    # print(training[0].shape)
    testing = np.asarray([X[0:,splitRatio:], Y[0:,splitRatio:]])
    # print(testing[0].shape)
    return training, testing

if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)

    a,b = split_data_into_training_and_testing([X,Y], 50)
    # W = train_weight_vector(X, Y, .001)  # arbitrary lambda for now
