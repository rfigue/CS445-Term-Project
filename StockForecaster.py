import pandas as pd
import numpy as np
import mlutilities as ml

def partition(X,T,trainFraction,shuffle=False,classification=False):
    # Skip the validation step
    validateFraction = 0
    testFraction = 1 - trainFraction
        
    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)
    
    if not classification:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        Xtrain = X[rowIndices[:nTrain],:]
        Ttrain = T[rowIndices[:nTrain],:]
        if nValidate > 0:
            Xvalidate = X[rowIndices[nTrain:nTrain+nValidate],:]
            Tvalidate = T[rowIndices[nTrain:nTrain:nValidate],:]
        Xtest = X[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        Ttest = T[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        
    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        trainIndices = []
        validateIndices = []
        testIndices = []
        for c in classes:
            # row indices for class c
            cRows = np.where(T[rowIndices,:] == c)[0]
            # collect row indices for class c for each partition
            n = len(cRows)
            nTrain = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate
            trainIndices += rowIndices[cRows[:nTrain]].tolist()
            if nValidate > 0:
                validateIndices += rowIndices[cRows[nTrain:nTrain+nValidate]].tolist()
            testIndices += rowIndices[cRows[nTrain+nValidate:nTrain+nValidate+nTest]].tolist()
        Xtrain = X[trainIndices,:]
        Ttrain = T[trainIndices,:]
        if nValidate > 0:
            Xvalidate = X[validateIndices,:]
            Tvalidate = T[validateIndices,:]
        Xtest = X[testIndices,:]
        Ttest = T[testIndices,:]
    if nValidate > 0:
        return Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
    else:
        return Xtrain,Ttrain,Xtest,Ttest

!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Disney stock
disney = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DIS = pd.read_csv(disney, delimiter=',', engine = 'python')
DISX = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DIST = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DISX = np.array(DISX)
DIST = (np.array(DIST)).reshape(DIST.shape[0],-1)
DISXTrain, DISTTrain, DISXtest, DISTtest = partition(DISX, DIST, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
depot = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DEP = pd.read_csv(depot, delimiter=',', engine = 'python')
DEPX = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DEPT = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DEPX = np.array(DEPX)
DEPT = (np.array(DEPT)).reshape(DEPT.shape[0],-1)
DEPXTrain, DEPTTrain, DEPXtest, DEPTtest = partition(DEPX, DEPT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)
APPXTrain, APPTTrain, APPXtest, APPTtest = partition(APPX, APPT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Nike
nike = 'NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd'
NIK = pd.read_csv(nike, delimiter=',', engine = 'python')
NIKX = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
NIKT = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
NIKX = np.array(NIKX)
NIKT = (np.array(NIKT)).reshape(NIKT.shape[0],-1)
NIKXTrain, NIKTTrain, NIKXtest, NIKTtest = partition(NIKX, NIKT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Microsoft
micro = 'MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
MIC = pd.read_csv(micro, delimiter=',', engine = 'python')
MICX = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
MICT = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
MICX = np.array(MICX)
MICT = (np.array(MICT)).reshape(MICT.shape[0],-1)
MICXTrain, MICTTrain, MICXtest, MICTtest = partition(MICX, MICT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Intel
intel = 'INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
INT = pd.read_csv(intel, delimiter=',', engine = 'python')
INTX = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
INTT = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
INTX = np.array(INTX)
INTT = (np.array(INTT)).reshape(INTT.shape[0],-1)
INTXTrain, INTTTrain, INTXtest, INTTtest = partition(INTX, INTT, 0.8)


