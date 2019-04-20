import numpy as np 
import numpy.linalg as la
import csv
import sys
import os


def parseDatabaseIntoMatrix(inputFile):
    inputFile = os.path.abspath(inputFile)
    with open(inputFile) as csvfile:
        datareader = csv.reader(csvfile)
        data_point_matrix = []
        for row in datareader:
            data_point =[]
            for i in range(len(row)):
                data_point.append(row[i])
            data_point_matrix.append(data_point)
        return([np.asarray(data_point_matrix), np.transpose(data_point_matrix)])

if __name__ == '__main__':
    inputFile = sys.argv[1]
    dataset_and_tranpsose = parseDatabaseIntoMatrix(inputFile)
    print(dataset_and_tranpsose[0])
    print(dataset_and_tranpsose[1])



            