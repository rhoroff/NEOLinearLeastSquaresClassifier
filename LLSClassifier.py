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
            data_point_matrix = np.concatenate((data_point_matrix, ones),1)

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
            data_point_matrix = np.concatenate((data_point_matrix, ones),1)

            return data_point_matrix.T, np.asarray(classifications).T
    else:
        return [], []


def train_weight_vector(inputMatrix, classMatrix, inputLambda):
    X = np.asarray(inputMatrix)
    Y = np.asarray(classMatrix)
    # Arbitrarily small value to start, will vary for testing purposes
    conditioning_lambda = inputLambda
    W = np.dot(la.inv(np.dot(X, X.T) + conditioning_lambda), np.dot(X, Y.T))
    print(W)
    print(la.lstsq(X.T,Y.T)[0])

    return W

def split_data_into_training_and_testing(database, trainingPercentage):
    return 0


if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)
    W = train_weight_vector(X, Y, .001)  # arbitrary lambda for now
