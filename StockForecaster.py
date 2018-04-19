import pandas as pd
import numpy as np
import mlutilities as ml

!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Disney stock
fName = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
disney = pd.read_csv(fName, delimiter=',', engine = 'python')

!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
fName = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
depot = pd.read_csv(fName, delimiter=',', engine = 'python')

!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
fName = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
apple = pd.read_csv(fName, delimiter=',', engine = 'python')

!wget https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Nike
fName = 'NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd'
nike = pd.read_csv(fName, delimiter=',', engine = 'python')

