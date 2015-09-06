import csv as csv
import numpy as np

def readData(fileName):

	with open(fileName,'rb') as csvTrainFile:
		fileReader = csv.reader(csvTrainFile)
		next(fileReader)
		X = []
		Y = []
		for row in fileReader:
			Y.append(row[1])
			X.append(row[2:])
		return (np.array(X),np.array(Y,dtype=float))



def classifySurvival(trainFileName):
	(X,Y) = prepareData(trainFileName)
	Beta = np.random.rand(2,1)
	prob = np.vstack((1/(1 + np.exp(-(np.dot(X,Beta))))))
	gradient = np.dot(X.transpose(),(Y-prob))
	print(gradient)

def prepareData(trainFileName):	
	(X,Y) = readData(trainFileName)
	size = len(X)
	ModX = (np.array([1 if x=='male' else 0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	ModX = np.concatenate((ones,ModX),1)
	return (ModX,Y.reshape(size,1))





def predictOnTestData(testFileName, model):
	(X,Y) = readData(testFileName, model)


classifySurvival('train.csv')