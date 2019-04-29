import numpy as np
import numpy.linalg as la
import csv, sys, os, re, math


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
    elif classification == '0':
        return [0,1,0]


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
                class_col = 3
                classification = row[class_col]

                # '1' -> [1,0], '2' -> [0,1[]
                classification = classification_to_vector(row[3])
                classification.pop(2)
                classifications.append(classification)

                for i in range(1, len(row)):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)
            #Append a 1 to every data point for the free param
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            ones = np.ones((data_point_matrix.shape[0], 1))
            data_point_matrix = np.concatenate((data_point_matrix, ones), 1)

            return data_point_matrix.T, np.asarray(classifications).T
    elif re.search("cmc.data", inputFile):
        with open(inputFile) as csvfile:
            datareader = csv.reader(csvfile)

            # data_point_matrix[0] -> classifications[0]
            data_point_matrix = []
            classifications = []

            for row in datareader:
                data_point = []
                class_col = 9
                classification = row[class_col]

                # '1' -> [1,0,0], '2' -> [0,1,0], '3' -> [0,0,1]
                classifications.append(classification_to_vector(row[9]))

                for i in range(1, len(row)):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)
            #Append a 1 to every data point for the free param
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            data_point_matrix = np.asarray(data_point_matrix, dtype=np.float32)
            ones = np.ones((data_point_matrix.shape[0], 1))
            data_point_matrix = np.concatenate((data_point_matrix, ones), 1)

            return data_point_matrix.T, np.asarray(classifications).T
    elif re.search("haberman.data", inputFile):
        with open(inputFile) as csvfile:
            datareader = csv.reader(csvfile)

            # data_point_matrix[0] -> classifications[0]
            data_point_matrix = []
            classifications = []

            for row in datareader:
                data_point = []
                class_col = len(row)-1
                classification = row[class_col]

                # '1' -> [1,0,0], '2' -> [0,1,0], '3' -> [0,0,1]
                classifications.append(classification_to_vector(row[3]))

                for i in range(1, len(row)):
                    data_point.append(row[i])

                data_point_matrix.append(data_point)

            #Append a 1 to every data point for the free param
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
    (classes, numberOfEachClass) = np.unique(Y_t, return_counts=True, axis=0)
    training = [[], []]
    testing = [[], []]
    numElsInTraining = int(X.shape[1] * (.01*trainingPercentage))
    numElsPerClass = math.ceil(numElsInTraining / numberOfEachClass.shape[0])
    for classification in classes:
        counter = 0
        i = 0
        while i < Y_t.shape[0]:
            if (np.array_equal(Y_t[i], classification)) and counter < numElsPerClass:
                training[0].append(X_t[i])
                training[1].append(Y_t[i])
                X_t = np.delete(X_t, i, axis=0)
                Y_t = np.delete(Y_t, i, axis=0)
                i = 0
                counter = counter + 1
            else:
                i = i + 1

    training[0] = np.asarray(training[0]).T
    training[1] = np.asarray(training[1]).T
    # print(np.unique(training[1], return_counts=True, axis=1))
    testing = [X_t.T, Y_t.T]
    # print(training)
    # print(testing.shape[1])
    # print(testing)
    # training = [X[0:, :numElsInTraining], Y[0:, :numElsInTraining]]
    # testing = [X[0:, numElsInTraining:], Y[0:, numElsInTraining:]]
    return training, testing


def classify(W, x_i):
    """Classifies x_i as one of the classes in W by multiplying each W_k with x_i and taking the largest of the three to be the class that x_i belongs to
    
    Arguments:
        W (2D np.array) -- a weight vector, made using the Linear Least Squares Classification method, each column contains a series of weights corresponding to an input class
        x_i (1D np.array) -- a data point to be classified
    
    Returns:
        classificationofX_i (1D np.array) -- the classification that x_i is calculated to have belonged to based off of the weight vector W
    """
    classification = np.eye(W.shape[1])
    curMax = np.dot(W.T[0], x_i)
    curMaxIndex = 0
    i = 1
    for i in range(W.shape[1]):
        if np.dot(W.T[i], x_i) > curMax:  # Tranpsose makes things easier
            curMax = np.dot(W.T[i], x_i)
            curMaxIndex = i
    classificationOfX_i = classification.T[curMaxIndex].T

    return classificationOfX_i


def classify_set_of_data_points(W, X):
    """Classifies a whole set of data points against a weight vector W
    
    Arguments:
        W (2D np.array) -- the weight vector W used for containing class weights for data points 
        X (2D np.array) -- a vector of input data points to classify

    Returns: 
        classesOfX (2D np.array) -- a matrix of classification that maps one to one to the indices of the input matrix X
    """
    classesOfX = []
    X = X.T
    for i in range(X.shape[0]):  # Easier to work with tranposes
        classesOfX.append(classify(W, X[i]))
    # print(np.asarray(classesOfX))
    return np.asarray(classesOfX, dtype=int).T

def check_miscalculations(knownClasses, learnedClasses):
    numMisclassifications = 0
    for i in range(knownClasses.shape[1]):
        if not (np.array_equal(knownClasses.T[i], learnedClasses.T[i])) :
            numMisclassifications = numMisclassifications + 1
    return numMisclassifications

def vector_to_classification(vector):
    if vector == [1, 0, 0]:
        return '1'
    elif vector == [0, 1, 0]:
        return '2'
    elif vector == [0, 0, 1]:
        return '3'

# if __name__ == '__main__':
#     inputFile = sys.argv[1]
#     X, Y = parse_database_into_matrix(inputFile)
#     testingTrainingPercentage = [50,60, 70,80,90]
#     for percentage in testingTrainingPercentage:
#         C = split_data_into_training_and_testing([X, Y], percentage)
#         training_X = C[0][0]
#         training_Y = C[0][1]
#         testing_X = C[1][0]
#         testing_Y = C[1][1]

#         # arbitrary lambda for now
#         W = train_weight_vector(training_X, training_Y, .001)
#         tested_Y = classify_set_of_data_points(W, testing_X)
#         print(check_miscalculations(testing_Y, tested_Y))

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

def perctange_based_training(data, labels, percentage):
    C = split_data_into_training_and_testing([data, labels], percentage)
    training_X = C[0][0]
    training_Y = C[0][1]
    testing_X = C[1][0]
    testing_Y = C[1][1]

    # arbitrary lambda for now
    W = train_weight_vector(training_X, training_Y, .001)
    tested_Y = classify_set_of_data_points(W, testing_X)

    labels = np.asarray(list(vector_to_classification(v) for v in testing_Y.T.tolist()))
    predictions = np.asarray(list(vector_to_classification(v) for v in tested_Y.T.tolist()))

    print(classification_error(labels, predictions))    

if __name__ == '__main__':
    inputFile = sys.argv[1]
    X, Y = parse_database_into_matrix(inputFile)
    for p in [50,60,70,80,90]:
        perctange_based_training(X,Y,p)

    # print(check_miscalculations(testing_Y, tested_Y))
