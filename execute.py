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
	Beta = (1)*np.random.rand(2,1)
	prob = lambda Beta:1/(1 + np.exp(-(X.dot(Beta))))
	print(prob(Beta))
	gradient = lambda Beta: X.transpose().dot(Y-prob(Beta))
	W = lambda Beta:prob(Beta).dot((1-prob(Beta)).transpose())
	likelihood = lambda Beta: Y.transpose().dot(np.log(prob(Beta))) + (1-Y).transpose().dot(np.log(1-prob(Beta)))
	hessian = lambda Beta: -1*(X.transpose().dot(W(Beta))).dot(X)
	print(np.linalg.inv(hessian(Beta)))
	print(gradient(Beta))
	for i in range(1,4):
		try:
			diff = (np.linalg.inv(hessian(Beta))).dot(gradient(Beta))
		except:
			break;	
		Beta = Beta - diff
	print(Beta)

	#diff = (np.linalg.inv(hessian(Beta))).dot(gradient(Beta))
	#	diff = 0.1*gradient(Beta)
	#Beta = Beta - diff
	#print(Beta)
	print(prob(Beta))
	return Beta	

def prepareData(trainFileName):	
	(X,Y) = readData(trainFileName)
	size = len(X)
	ModX = (np.array([1.0 if x=='male' else 0.0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	ModX = np.concatenate((ones,ModX),1)
	return (ModX,Y.reshape(size,1))





def predictOnTestData(testFileName, model):

	with open(testFileName,'rb') as csvTrainFile:
		fileReader = csv.reader(csvTrainFile)
		next(fileReader)
		X = []
		for row in fileReader:
			X.append(row[1:])
		X = np.array(X)
		print(X)
	size = len(X)
	ModX = (np.array([1.0 if x=='male' else 0.0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	ModX = np.concatenate((ones,ModX),1)
	X = ModX

	prob = lambda Beta:1/(1 + np.exp(-(X.dot(Beta))))

	print(prob(model))


model = classifySurvival('train.csv')
predictOnTestData('test.csv',model)