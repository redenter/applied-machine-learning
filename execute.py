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
	prob = lambda Beta:np.vstack((1/(1 + np.exp(-(np.dot(X,Beta))))))
	gradient = lambda Beta:np.dot(X.transpose(),(Y-prob(Beta)))
	W = lambda Beta:prob(Beta).dot((1-prob(Beta)).transpose())

	hessian = lambda Beta: -1*(X.transpose().dot(W(Beta))).dot(X)

	for i in range(1,20):
		Beta = Beta - hessian(Beta).dot(gradient(Beta))

	return Beta	

def prepareData(trainFileName):	
	(X,Y) = readData(trainFileName)
	size = len(X)
	ModX = (np.array([1 if x=='male' else 0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	ModX = np.concatenate((ones,ModX),1)
	return (ModX,Y.reshape(size,1))





def predictOnTestData(testFileName, model):
	(X,Y) = prepareData(testFileName)
	prob = np.vstack((1/(1 + np.exp(-(np.dot(X,model))))))
	result = [1 if x>0.5 else 0 for x in prob]
	print(result)



model = classifySurvival('train.csv')
predictOnTestData('test.csv',model)