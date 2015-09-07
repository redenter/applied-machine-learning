import csv as csv
import numpy as np



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
	Beta = np.random.rand(2,1)
	prob = lambda Beta:1/(1 + np.exp(-(X.dot(Beta))))
	gradient = lambda Beta: X.transpose().dot(Y-prob(Beta))
	W = lambda Beta:prob(Beta).dot((1-prob(Beta)).transpose())
	likelihood = lambda Beta: Y.transpose().dot(np.log(prob(Beta))) + (1-Y).transpose().dot(np.log(1-prob(Beta)))
	hessian = lambda Beta: -1*(X.transpose().dot(W(Beta))).dot(X)
	

	for i in range(1,20):
		try:
			diff = (np.linalg.inv(hessian(Beta))).dot(gradient(Beta))
		except:
			break;
		Beta = Beta - diff
		print(likelihood(Beta))

	return Beta	




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
	prob = 1/(1 + np.exp(-(X.dot(model))))
	return prob

def writeResultToFile(Ids,result,fileName):
	resp = np.concatenate((Ids,result),1)	
	np.savetxt(fileName,resp,delimiter = ',',fmt='%10.0f')

X,Y = prepareTrainData('train.csv')
model = logisticRegression(X,Y)
Ids,X = prepareTestData('test.csv')

result = predictOnTestData(X,model)
writeResultToFile(Ids,result,'result.csv')