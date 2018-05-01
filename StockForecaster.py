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

!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
depot = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DEP = pd.read_csv(depot, delimiter=',', engine = 'python')
DEPX = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DEPT = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DEPX = np.array(DEPX)
DEPT = (np.array(DEPT)).reshape(DEPT.shape[0],-1)

!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)

!wget https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Nike
fName = 'NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd'
nike = pd.read_csv(fName, delimiter=',', engine = 'python')

!wget https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Microsoft
fName = 'MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
micro = pd.read_csv(fName, delimiter=',', engine = 'python')

!wget https://www.quandl.com/api/v3/datasets/EOD/INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Intel
fName = 'INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
intel = pd.read_csv(fName, delimiter=',', engine = 'python')
