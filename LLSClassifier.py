import numpy as np
import numpy.linalg as la
import csv
import sys
import os
import re


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
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            ones = np.ones((data_point_matrix.shape[0], 1))
            data_point_matrix = np.concatenate((data_point_matrix, ones), 1)

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
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            ones = np.ones((data_point_matrix.shape[0], 1))
            data_point_matrix = np.concatenate((data_point_matrix, ones), 1)

            return data_point_matrix.T, np.asarray(classifications).T
    else:
        return [], []


def train_weight_vector(inputMatrix, classMatrix, testingLambda):
    """Trains a weight vector based off an input matrix, varying values of lambda to minimize misclassification errors

    Args:\n
        inputMatrix (2D list): The matrix containing input data points as columns and featrues as rows\n
        classMatrix (2D list): The matrix containing classifications for the inputMatrix, maps one to one based on index\n 
        testingLambda (float): The lambda value to be used for conditioning the weight matrix W\n

    Returns:\n
        2D numpy.array: The weight matrix W\n
            
    """

    X = np.asarray(inputMatrix)
    Y = np.asarray(classMatrix)

    # Arbitrarily small value to start, will vary for testing purposes
    conditioning_lambda = testingLambda
    W = np.dot(la.inv(np.dot(X, X.T) +
                      (conditioning_lambda * np.eye(X.shape[0]))), np.dot(X, Y.T))
    return W


def split_data_into_training_and_testing(database, trainingPercentage):
    """Splits data points into training and testing sets based off of the passed in training percentage
    Args:\n 
        database (list of np.arrays): A list containing an array of data points as its first entry and an array of classification for the data points as its second entry.The indices of these arrays map one to one\n 
        trainingPercentage (int): The percentage of testing values to split the data points and their classifications into\n

    Returns:\n 
        list of list of np.arrays: A list containing two lists, the first the training set and its associated labels and the second the testing set and its associated labels\n
    """
    X = np.asarray(database[0])
    Y = np.asarray(database[1])
    #Easier to work with tranposed data
    X_t = X.T 
    Y_t = Y.T 
    (classes, numberOfEachClass) = np.unique(Y_t, return_counts=True, axis = 0)
    training = [[], []]
    testing = [[],[]]
    numElsInTraining = int(X.shape[1] * (.01*trainingPercentage))
    numElsPerClass = int(numElsInTraining / numberOfEachClass.shape[0])
    for classification in classes:
        counter = 0
        for i in range(Y_t.shape[0] - 1):
            if (np.array_equal(Y_t[i], classification)) and counter < numElsPerClass:
                training[0].append(X_t[i])
                training[1].append(Y_t[i])
                np.delete(X_t, i, axis = 0)
                np.delete(Y_t,i, axis = 0)
                i = 0
                counter = counter + 1

    training[0] = np.asarray(training[0]).T
    training[1] = np.asarray(training[1]).T
    print(training[0].shape[1])
    print(training[1].shape[1])
    print(np.unique(training[1], return_counts=True, axis = 1))
    testing = [X_t.T, Y_t.T]
    print(testing[0].shape[1])
    print(testing[0].shape[1])
    # print(training)
    # print(testing.shape[1])
    # print(testing)
    # training = [X[0:, :numElsInTraining], Y[0:, :numElsInTraining]]
    # testing = [X[0:, numElsInTraining:], Y[0:, numElsInTraining:]]
    return training, testing


if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)
    C = split_data_into_training_and_testing([X, Y], 50)
    # training_X = C[0][0]
    # training_Y = C[0][1]
    # testing_X = C[1][0]
    # testing_Y = C[1][1]

    # arbitrary lambda for now
    # W = train_weight_vector(training_X, training_Y, .001)
    print(W)
