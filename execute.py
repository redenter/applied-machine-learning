import csv as csv
import numpy as np
from sklearn import preprocessing, datasets, linear_model,cross_validation
from collections import defaultdict


def getDataInDict(fileName):

	dataDict = defaultdict(list)
	with open(fileName) as f:
	    reader = csv.DictReader(f)
	    for row in reader: 
	        for (k,v) in row.items(): 
	        	dataDict[k].append(v) 

	return dataDict

def chooseFeatures(dataDict,ageAv = None):
	size = len(dataDict['Name'])
	Z = np.ones([size,3])
	Z[:,1] = (np.array([1.0 if x=='male' else 0.0 for x in dataDict['Sex']]))
	#Z[:,2] = dataDict['Pclass']
	#Z[:,3] = np.array([0 if x=='' else float(x) for x in dataDict['Fare']])
	#fillFare(dataDict)
	#Z[:,3] = dataDict['SibSp']
	#Z[:,4] = dataDict['Parch']
	if(ageAv is None):
		ageAv = createAgeDict(dataDict)

	Z[:,2] = fillAge(dataDict,ageAv)
#	Z[:,6] = np.array([1 if x=='C' else -1 if x=='Q' else 1 for x in dataDict['Embarked']])

	return Z,ageAv



def fillFare(dataDict):

	clMean = {}
	Pclass = np.array(dataDict['Pclass'],dtype= float)
	Fare = 	np.array([0 if x=='' else float(x) for x in dataDict['Fare']])

	clMean[1] = np.mean([y  for x,y in zip(Pclass,Fare) if x==1 and y<>0])
	clMean[2] = np.mean([y  for x,y in zip(Pclass,Fare) if x==2 and y<>0])
	clMean[3] = np.mean([y  for x,y in zip(Pclass,Fare) if x==3 and y<>0])

	for i in range(0,len(Fare)):
		if(Fare[i]==0):
			for key in clMean.keys():
				if(Pclass[i] == key):
					Fare[i] = clMean[key]	
	print Fare
	return Fare				

def createAgeDict(dataDict):
	size = len(dataDict['Name'])
	Names = dataDict['Name']
	Sex = dataDict['Sex']
	Pclass = np.array(dataDict['Pclass'],dtype=float)
	Age = np.array([-1 if x=='' else float(x) for x in dataDict['Age']])

	ageAv = {}	

	avgMr = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Mr.')<>-1 and y<>-1])
	ageAv['Mr.'] = avgMr

	avgMrs = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Mrs.')<>-1 and y<>-1])
	ageAv['Mrs.'] = avgMrs

	avgMast = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Master.')<>-1 and y<>-1])
	ageAv['Master.'] = avgMast

	avgCol = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Col.')<>-1 and y<>-1])
	ageAv['Col'] = avgCol

	avgDr = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Dr.')<>-1 and y<>-1])
	ageAv['Dr'] = avgDr

	avgMaleRestArr = ([y for x,y,z in zip(Names,Age,Sex) if str.find(x,'Mr.')==-1 and str.find(x,'Dr.')==-1 and str.find(x,'Master.')==-1 and str.find(x,'Rev.')==-1 and str.find(x,'Col.')==-1 and z=='male' and y<>-1])
	ageAv['MRest.'] = np.mean(avgMaleRestArr) if avgMaleRestArr<>[] else avgMr

	avgFeRestArr = ([y for x,y,z in zip(Names,Age,Sex) if str.find(x,'Mrs.')==-1 and str.find(x,'Miss.')==-1 and z=='female' and y<>-1])
	ageAv['Ferest.'] = np.mean(avgFeRestArr) if avgFeRestArr<>[] else (avgMrs+avgMissCl1+avgMissCl2+avgMissCl3)/4

	avgRev = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Rev.')<>-1 and y<>-1])
	ageAv['Rev'] = avgRev
	#avgMiss = np.mean([y for x,y in zip(Names,Age) if str.find(x,'Miss.')<>-1 and y<>-1])
	avgMissCl1 = np.mean([y for x,y,z in zip(Names,Age,Pclass) if str.find(x,'Miss.')<>-1 and y<>-1 and z==1])
	ageAv['Miss.1'] = avgMissCl1

	avgMissCl2 = np.mean([y for x,y,z in zip(Names,Age,Pclass) if str.find(x,'Miss.')<>-1 and y<>-1 and z==2])
	ageAv['Miss.2'] = avgMissCl2

	avgMissCl3 = np.mean([y for x,y,z in zip(Names,Age,Pclass) if str.find(x,'Miss.')<>-1 and y<>-1 and z==3])
	ageAv['Miss.3'] = avgMissCl3

	return ageAv


def fillAge(dataDict,ageAv):
	Names = dataDict['Name']
	size = len(Names)
	Sex = dataDict['Sex']
	Pclass = np.array(dataDict['Pclass'],dtype=float)
	Age = np.array([-1 if x=='' else float(x) for x in dataDict['Age']])	

	for i in range(0,size):

		if(Age[i] ==-1):
			for key in ageAv.keys():
				if(str.find(Names[i],key)<>-1 or str.find(Names[i],key+str(Pclass[i]) )):
					Age[i] = ageAv[key]
					break
			if(Age[i]==-1):
				if(Sex[i]=='male'):
					Age[i] = ageAv['Rest.']		
				else:
					Age[i] = ageAv['FeRest.']	


	return Age		

def logisticRegression(X,Y):
	cls = linear_model.LogisticRegression()
	#Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X,Y)
	#cls.fit(Xtrain, Ytrain)
	cls.fit(X,Y)
	#print cls.score(Xtest, Ytest)
	return cls


def writeResultToFile(Ids,result,Sex,Pclass,fileName):
	f = open(fileName,'wb')
	ob= csv.writer(f)
	ob.writerow(['PassengerId','Survived'])
#	ob.writerows(zip(Ids,result,Sex,Pclass))
	ob.writerows(zip(Ids,result))
	f.close()
	


##### Training ######

trainDataDict = getDataInDict('train.csv')
X,ageAv = chooseFeatures(trainDataDict)
Y = trainDataDict['Survived']
cls = logisticRegression(X,Y)
print cls

#### Testing #####
testDataDict = getDataInDict('test.csv')
X,ageAv = chooseFeatures(testDataDict,ageAv)
Ids = np.array(testDataDict['PassengerId'])
Sex = testDataDict['Sex']
Pclass = testDataDict['Pclass']
result = cls.predict(X)

#### Result submission #####
writeResultToFile(Ids,result,Sex,Pclass,'result-genderAge.csv')



