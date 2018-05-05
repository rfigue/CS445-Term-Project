import pandas as pd
import numpy as np
import mlutilities as ml

!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Disney stock
disney = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DIS = pd.read_csv(disney, delimiter=',', engine = 'python')
DISX = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DIST = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DISX = np.array(DISX)
DIST = (np.array(DIST)).reshape(DIST.shape[0],-1)
DISXTrain, DISTTrain, DISXtest, DISTtest = ml.partition(DISX, DIST, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
depot = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DEP = pd.read_csv(depot, delimiter=',', engine = 'python')
DEPX = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DEPT = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DEPX = np.array(DEPX)
DEPT = (np.array(DEPT)).reshape(DEPT.shape[0],-1)
DEPXTrain, DEPTTrain, DEPXtest, DEPTtest = ml.partition(DEPX, DEPT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)
APPXTrain, APPTTrain, APPXtest, APPTtest = ml.partition(APPX, APPT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Nike
nike = 'NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd'
NIK = pd.read_csv(nike, delimiter=',', engine = 'python')
NIKX = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
NIKT = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
NIKX = np.array(NIKX)
NIKT = (np.array(NIKT)).reshape(NIKT.shape[0],-1)
NIKXTrain, NIKTTrain, NIKXtest, NIKTtest = ml.partition(NIKX, NIKT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Microsoft
micro = 'MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
MIC = pd.read_csv(micro, delimiter=',', engine = 'python')
MICX = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
MICT = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
MICX = np.array(MICX)
MICT = (np.array(MICT)).reshape(MICT.shape[0],-1)
MICXTrain, MICTTrain, MICXtest, MICTtest = ml.partition(MICX, MICT, 0.8)

!wget https://www.quandl.com/api/v3/datasets/EOD/INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Intel
intel = 'INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
INT = pd.read_csv(intel, delimiter=',', engine = 'python')
INTX = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
INTT = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
INTX = np.array(INTX)
INTT = (np.array(INTT)).reshape(INTT.shape[0],-1)
INTXTrain, INTTTrain, INTXtest, INTTtest = ml.partition(INTX, INTT, 0.8)


