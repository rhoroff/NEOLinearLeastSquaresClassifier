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


def compute_class_matrix(inputMatrix, classDiscriminatorColumn):
    # Check if the descirminators are already in int form
    if isinstance(inputMatrix[0][classDiscriminatorColumn], int):
        for i in range(inputMatrix.shape):

    else:



def compute_weight_vector(inputMatrix, classMatrix):
    return


if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, X_t = parse_database_into_matrix(inputFile)
    print(X)
    print(X_t)
