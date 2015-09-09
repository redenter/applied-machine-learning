import csv as csv
import numpy as np
from sklearn import preprocessing, datasets, linear_model,cross_validation



def prepareTrainData(trainFileName):

	X = []
	Y = []
	with open(trainFileName,'rb') as csvTrainFile:
		fileReader = csv.reader(csvTrainFile)
		next(fileReader)
		for row in fileReader:
			Y.append(row[1])
			X.append(row[2:])

	X = np.array(X)
	size = len(X)
	ModX = (np.array([1.0 if x=='male' else 0.0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	X = np.concatenate((ones,ModX),1)

	return (X,(np.array(Y,dtype=float)).reshape(size,1))




def logisticRegression(X,Y):
	cls = linear_model.LogisticRegression()
	Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X,Y)
	cls.fit(Xtrain, Ytrain)

	print(cls.predict(Xtest))
	print cls.score(Xtest, Ytest)
	return cls


def prepareTestData(testFileName):
	
	X = []
	Ids = []
	with open(testFileName,'rb') as csvFile:
		fileReader = csv.reader(csvFile)
		next(fileReader)
		for row in fileReader:
			Ids.append(row[0])
			X.append(row[1:])
		X = np.array(X)

	size = len(X)
	ModX = (np.array([1.0 if x=='male' else 0.0 for x in X[:,2]])).reshape(size,1)
	ones = np.vstack(np.ones(size))
	ModX = np.concatenate((ones,ModX),1)
	
	return np.array(Ids,dtype=float).reshape(size,1),ModX		



def predictOnTestData(X, model):
	result = cls.predict(X)
	return result.reshape(len(result),1)

def writeResultToFile(Ids,result,fileName):
	
	resp = np.concatenate((Ids,result),1)	
	np.savetxt(fileName,resp,delimiter = ',',fmt='%10.0f')

X,Y = prepareTrainData('train.csv')
cls = logisticRegression(X,Y)
Ids,X = prepareTestData('test.csv')

result = predictOnTestData(X,cls)
writeResultToFile(Ids,result,'result.csv')