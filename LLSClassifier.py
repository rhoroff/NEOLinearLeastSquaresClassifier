import numpy as np
import numpy.linalg as la
import csv, sys, os, re

def classification_to_vector(classification):
    if classification == '1':
        return [1,0,0]
    elif classification == '2':
        return [0,1,0]
    elif classification == '3':
        return [0,0,1]
    elif classification == 'Iris-setosa':
        return [1,0,0]
    elif classification == 'Iris-virginica':
        return [0,1,0]
    elif classification == 'Iris-versicolor':
        return [0,0,1]


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
                classifications.append(classification_to_vector(classification))
                for i in range(len(row)-1):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            data_point_matrix = data_point_matrix.T
            
            ones = np.ones((1, data_point_matrix.shape[1]))
            data_point_matrix = np.concatenate((data_point_matrix, ones))
            data_point_matrix = data_point_matrix.T
            print(data_point_matrix)

            return data_point_matrix, classifications

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

                for i in range(1,len(row)):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            data_point_matrix = data_point_matrix.T
            
            ones = np.ones((1, data_point_matrix.shape[1]))
            data_point_matrix = np.concatenate((data_point_matrix, ones))
            data_point_matrix = data_point_matrix.T

            return data_point_matrix, np.asarray(classifications)
    else:
        return [],[]

def train_weight_vector(inputMatrix, classMatrix, inputLambda):
    X = np.asarray(inputMatrix)
    X = X.T
    Y = np.asarray(classMatrix)
    conditioning_lambda = inputLambda #Arbitrarily small value to start, will vary for testing purposes
    W = np.dot(la.inv(np.dot(X,X.T) + conditioning_lambda), np.dot(X,Y))
    calc_W = (np.asarray(la.lstsq(X.T, Y)[0]))

    return W

def classify(W,x_i):
    return 1

if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)
    W = train_weight_vector(X,Y, .001) # arbitrary lambda for now
