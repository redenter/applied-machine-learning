import numpy as np
import matplotlib.pyplot as plt
import csv


def readData(fileName):

	with open(fileName,'rb') as csvTrainFile:
		fileReader = csv.reader(csvTrainFile)
		next(fileReader)
		X = []
		Y = []
		for row in fileReader:
			Y.append(row[0])
			X.append(row[1:])
		return (np.array(X,dtype = float),np.array(Y))	

def display(inputVec):
	
	inputVec = np.reshape(inputVec,(28,28))
	plt.imshow(inputVec,cmap=plt.cm.gray)
	plt.show()


(X,Y)=readData('train.csv')
display(X[431])