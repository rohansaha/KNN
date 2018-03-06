"""==============================================================
 COURSE:	CSC 635, Homework 2
 PROGRAMMER:	Rohan Saha
 Trace ID:      Rohan2728
 DATE:	        3/6/2018
 DESCRIPTION:   To create a KNN algorithm, use MINIST_test.csv to
                test the algorithm and MINIST_train.csv to train
                the algorithm.
 FILES:	        KNN.py
 DATASET:       MNIST_train.csv, MNIST_test.csv
 =============================================================="""

#---------------------------------Imports--------------------------------------      
from csv import reader
from math import pow, sqrt
from random import shuffle, randint
from operator import itemgetter
#------------------------------------------------------------------------------

#---------------------------------Functions------------------------------------

'''Normalizes the data set'''
def normalize(data):
    result = list()
    for index, row in enumerate(data):
        row_denom = max(row) -  min(row)
        temp= list()
        for i, value in enumerate(row):
            if i != 0:
                n_data = (value - min(row)) / row_denom
                temp.append(n_data)
            else:
                temp.append(value)
        result.append(temp)
    return result

'''Calculates the distance between two attribute values'''
def eucildean(testData, trainingData):
    distance= 0
    for i in range(len(testData)):
        distance +=pow(testData[i] - trainingData[i],2)
    return sqrt(distance)

'''KNN Algorithm'''
def KNN(testData, trainingData, k=3) :
    prediction= list()
    print('\n\nK = ', k)
    for testSample in testData:
        distance = list()
        for trainingSample in trainingData:
            distance.append([trainingSample[0], eucildean(testSample[1:], trainingSample[1:])])
        distance.sort(key=itemgetter(1))
        neighbors = [distance[d] for d in range(k)]
        result = getVotes(neighbors)
        prediction.append(result)
        print('Desired Class = ', testSample[0], '\tComputed Class = ',result)
    accuracy_result = accuracy(testData, prediction)
    print ('Accuracy Rate = ', accuracy_result[0])
    print('Number of mis-classified test samples = ', accuracy_result[1])
    print('Total number of test samples = ', len(testData))

'''Gets the weighted votes of the nearest neighbours'''
def getVotes(neighbors):
    vote = 0
    for n in neighbors:
        if n[1] == 0:
            vote = 0
        else:
            vote = 1/n[1]
        n.append(vote)
    neighbors.sort(key=itemgetter(2))
    return neighbors[0][0]

'''Calculates the accuracy of the KNN algorithm predicted set'''
def accuracy(testSet, predictedSet):
    correct = 0
    misses = 0
    for i in range(len(testSet)):
        if testSet[i][0] == predictedSet[i]:
            correct +=1
        else:
            misses +=1
    return [round((correct/ float(len(predictedSet)))*100,5), misses]

'''Reads the data from csv file and normalize the dataset'''
def dataPreprocessing(filename, isTestData = True):
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        data = list(csv_reader)
    del(data[0])
    data = [list(map(int, d)) for d in data]
    if not isTestData:
        shuffle(data)
    #data = normalize(data)
    return data
#------------------------------------------------------------------------------

#---------------------------------Program Main---------------------------------

if __name__ == '__main__':
    trainingFilename = '.\\dataset\\MNIST_train.csv'
    testFilename = '.\\dataset\\MNIST_test.csv'
    trainingData = dataPreprocessing(trainingFilename, False)
    testingData = dataPreprocessing(testFilename)
    
    #The highest accuracy result is obtained for k = 4
    k  = 4
    
    # Get K values
    # t_Data =  trainingData.copy()
    # for k in range(1, 11):
    #     randIndex = randint(1, len(t_Data) - 1)
    #     kSample = [t_Data.pop(randIndex)]
    #     KNN(kSample, t_Data, k)
    #for k in range(1,11):
    KNN(testingData,trainingData, k)

#---------------------------------End of Program-------------------------------
