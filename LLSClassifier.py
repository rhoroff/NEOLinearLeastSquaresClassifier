import numpy as np
import numpy.linalg as la
import csv
import sys
import os



def parse_database_into_matrix(inputFile):
    # Full class path in case its necessary
    inputFile = os.path.abspath(inputFile)
    with open(inputFile) as csvfile:
        datareader = csv.reader(csvfile)
        data_point_matrix = []
        for row in datareader:
            data_point = []
            for i in range(len(row)):
                data_point.append(row[i])
            data_point_matrix.append(data_point)
        return np.asarray(data_point_matrix), np.asarray(data_point_matrix).T


def train_weight_vector(inputMatrix, classMatrix, inputLambda):
    X = np.asarray(inputMatrix)
    Y = np.asarray(classMatrix)
    conditioning_lambda = inputLambda #Arbitrarily small value to start, will vary for testing purposes
    la.inv(X * X.T + conditioning_lambda)*(X*Y.T)
    return la.inv(X * X.T + conditioning_lambda)*(X*Y.T)

if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)

