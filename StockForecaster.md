CS 445: Using Time Embedding to Forecast Stock Behavior
Team Members: Rodolfo Figueroa, Evan Salzman
Introduction
In this report we will discuss the process of forecasting stock prices by referencing their past behavior. In order to accomplish this, the team has implemented code to use time embedding to record the changes of six highlighted stocks over many years. The stocks that will be used throughout this investigation include: Disney (DIS), Apple (AAPL), Home Depot (HD), Nike (NKE), Intel (INTC) and Microsoft (MSFT) from the NYSE (New York Stock Exchange). The CSV files containing the data from these stocks were found at: https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices.

Description of Datasets
The data sets for the six stocks chosen for this project are DIS, AAPL, HD, NKE, INTC and MSFT from the NYSE. Each of these files is downloaded in this notebook using pandas from the Quandl site providing End of Day US Stock Prices. Each .csv file contains the following attributes:

Date (1.2.1962 - 4.19.2018)
Open (decimal value of the price of the stock when it opened on the given day)
High (decimal value the peak in price of the stock for the given day)
Low (decimal value the lowest price of the stock for the given day)
Close (decimal value the final price of the stock from the market closed on that the given day)
Volume (a value to show the amount of traffic the stock experience on the given day)
Dividend (a value to show the unadjusted dividend on any ex-dividend if applicable, else 0.0.)
Split (a value to show any split that occurred on the given DATE, else 1.0)
Adj_Open (adjusted decimal value of the price of the stock when it opened on the given day)
Adj_High (adjusted decimal value of the peak in price of the stock for the given day)
Adj_Low (adjusted decimal value of the lowest price of the stock for the given day)
Adj_Close (adjusted decimal value of the final price of the stock from when the market closed on the given day)
Adj_Volume (an adjusted value to show the amount of traffic the stock experience on the given day)
The values that have been adjusted now reflect any dividends and splits that occurred on each specific day. The values that are adjusted are done with these respective calculations detailed at the following link (http://www.crsp.com/products/documentation/crsp-calculations):

Price and dividend data are adjusted with the calculation:

    A(t) = P(t) / C(t),

    where A(t) is the adjusted value at time t, P(t) is the raw value at time t, and C(t) is the cumulative 
    adjustment factor at time t.

Share and volume data are adjusted with the calculation:

    A(t) = P(t) * C(t),

    where A(t) is the adjusted value at time t, P(t) is the raw value at time t, and C(t) is the cumulative 
    adjustment factor at time t.

For the team's purposes in this assignment, the adjusted values will be used as they are more representative of how the stocks are performing in the marketing. From these "Adj" values, the Open, High and Low will be the input values, stored in the X matrix, while the Close will be the one and only target value in the T matrix. The stock price at the end of the day is the most important to consider as it is a reflection of the entire day, as it relates to how well the stock is performing compared to other days in its past and in comparison to other stocks. Through using these datasets and setting up the data in this way. it is possible to predict how each respective stock will perform in the market.

Introducion to Time Embedding
Take a sample and embed it in time using past samples. By using enough samples from a number of days of a given stock's performance in the stock market, it was possible for the team to predict the future behavior of each respective stock.

Experimentation with Stock Data and Prices
import pandas as pd
import numpy as np
import mlutilities as ml
#import neuralnetworksA4 as nn
!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
--2018-05-01 10:05:19--  https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
Resolving www.quandl.com (www.quandl.com)... 2400:cb00:2048:1::6819:3567, 2400:cb00:2048:1::6819:3667, 104.25.53.103, ...
Connecting to www.quandl.com (www.quandl.com)|2400:cb00:2048:1::6819:3567|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/csv]
Saving to: ‘DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd.4’

DIS.csv?api_key=q74     [  <=>               ]   1.87M  8.81MB/s    in 0.2s    

2018-05-01 10:05:25 (8.81 MB/s) - ‘DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd.4’ saved [1965115]

disney = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DIS = pd.read_csv(disney, delimiter=',', engine = 'python')
DISX = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DIST = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DISX = np.array(DISX)
DIST = (np.array(DIST)).reshape(DIST.shape[0],-1)
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
!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
--2018-04-19 15:22:08--  https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
Resolving www.quandl.com (www.quandl.com)... 104.25.53.103, 104.25.54.103
Connecting to www.quandl.com (www.quandl.com)|104.25.53.103|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/csv]
Saving to: ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd’

HD.csv?api_key=q74T     [  <=>               ]   1.22M  4.34MB/s    in 0.3s    

2018-04-19 15:22:11 (4.34 MB/s) - ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd’ saved [1279539]

HD = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd&start_date=1962-01-02'
H = pd.read_csv(disney, delimiter=',', engine = 'python')
H
Date	Open	High	Low	Close	Volume	Dividend	Split	Adj_Open	Adj_High	Adj_Low	Adj_Close	Adj_Volume
0	2018-04-19	101.000	101.5799	100.2200	100.89	6582208.0	0.0	1.0	101.000000	101.579900	100.220000	100.890000	6582208.00
1	2018-04-18	102.670	102.7100	101.2000	101.21	5820999.0	0.0	1.0	102.670000	102.710000	101.200000	101.210000	5820999.00
2	2018-04-17	101.200	102.5900	100.7500	102.17	9577287.0	0.0	1.0	101.200000	102.590000	100.750000	102.170000	9577287.00
3	2018-04-16	100.690	101.0000	99.7300	100.24	10327042.0	0.0	1.0	100.690000	101.000000	99.730000	100.240000	10327042.00
4	2018-04-13	101.000	101.5200	100.1600	100.35	6324346.0	0.0	1.0	101.000000	101.520000	100.160000	100.350000	6324346.00
5	2018-04-12	101.420	101.5100	99.6800	100.39	7331660.0	0.0	1.0	101.420000	101.510000	99.680000	100.390000	7331660.00
6	2018-04-11	100.780	101.6500	100.4100	100.80	6304786.0	0.0	1.0	100.780000	101.650000	100.410000	100.800000	6304786.00
7	2018-04-10	100.920	101.5250	100.3200	101.37	8147660.0	0.0	1.0	100.920000	101.525000	100.320000	101.370000	8147660.00
8	2018-04-09	100.700	101.5100	99.5800	99.70	7094275.0	0.0	1.0	100.700000	101.510000	99.580000	99.700000	7094275.00
9	2018-04-06	101.630	102.1900	99.4500	100.35	7161279.0	0.0	1.0	101.630000	102.190000	99.450000	100.350000	7161279.00
10	2018-04-05	101.360	102.3800	100.9800	102.11	6610132.0	0.0	1.0	101.360000	102.380000	100.980000	102.110000	6610132.00
11	2018-04-04	98.430	101.1500	97.7600	100.95	8759021.0	0.0	1.0	98.430000	101.150000	97.760000	100.950000	8759021.00
12	2018-04-03	98.800	99.4900	97.7000	99.42	8481249.0	0.0	1.0	98.800000	99.490000	97.700000	99.420000	8481249.00
13	2018-04-02	100.180	100.9900	97.7800	98.66	8203693.0	0.0	1.0	100.180000	100.990000	97.780000	98.660000	8203693.00
14	2018-03-29	99.010	101.2800	98.8600	100.44	9343626.0	0.0	1.0	99.010000	101.280000	98.860000	100.440000	9343626.00
15	2018-03-28	99.500	100.0800	98.1500	98.54	9099663.0	0.0	1.0	99.500000	100.080000	98.150000	98.540000	9099663.00
16	2018-03-27	100.880	101.1800	98.8900	99.36	7193524.0	0.0	1.0	100.880000	101.180000	98.890000	99.360000	7193524.00
17	2018-03-26	99.860	100.7800	99.0782	100.65	7340071.0	0.0	1.0	99.860000	100.780000	99.078200	100.650000	7340071.00
18	2018-03-23	100.850	101.1100	98.4500	98.54	7505075.0	0.0	1.0	100.850000	101.110000	98.450000	98.540000	7505075.00
19	2018-03-22	101.290	101.6400	100.4100	100.60	8911556.0	0.0	1.0	101.290000	101.640000	100.410000	100.600000	8911556.00
20	2018-03-21	101.500	102.9400	101.4200	101.82	6136431.0	0.0	1.0	101.500000	102.940000	101.420000	101.820000	6136431.00
21	2018-03-20	101.540	102.1500	100.7500	101.35	8330335.0	0.0	1.0	101.540000	102.150000	100.750000	101.350000	8330335.00
22	2018-03-19	102.800	102.9500	101.0000	101.48	6528320.0	0.0	1.0	102.800000	102.950000	101.000000	101.480000	6528320.00
23	2018-03-16	103.560	104.2700	102.8400	102.87	10467983.0	0.0	1.0	103.560000	104.270000	102.840000	102.870000	10467983.00
24	2018-03-15	104.000	104.2850	103.2400	103.24	5174578.0	0.0	1.0	104.000000	104.285000	103.240000	103.240000	5174578.00
25	2018-03-14	104.530	104.6500	103.4700	103.90	6235877.0	0.0	1.0	104.530000	104.650000	103.470000	103.900000	6235877.00
26	2018-03-13	105.830	105.8300	103.4200	103.73	6716539.0	0.0	1.0	105.830000	105.830000	103.420000	103.730000	6716539.00
27	2018-03-12	104.715	105.9400	104.7150	105.17	6484769.0	0.0	1.0	104.715000	105.940000	104.715000	105.170000	6484769.00
28	2018-03-09	104.430	104.7500	103.6425	104.73	5618412.0	0.0	1.0	104.430000	104.750000	103.642500	104.730000	5618412.00
29	2018-03-08	103.990	104.5850	103.4700	104.03	6626981.0	0.0	1.0	103.990000	104.585000	103.470000	104.030000	6626981.00
...	...	...	...	...	...	...	...	...	...	...	...	...	...
14142	1962-02-12	39.500	39.6200	39.2500	39.50	1598.0	0.0	1.0	0.066266	0.066468	0.065847	0.066266	311418.24
14143	1962-02-09	39.380	39.7500	39.3800	39.50	2597.0	0.0	1.0	0.066065	0.066686	0.066065	0.066266	506103.36
14144	1962-02-08	38.750	39.6200	38.7500	39.25	5295.0	0.0	1.0	0.065422	0.066891	0.065422	0.066266	1031889.60
14145	1962-02-07	38.000	38.5000	38.0000	38.25	3596.0	0.0	1.0	0.056428	0.057171	0.056428	0.056800	700788.48
14146	1962-02-06	38.000	38.2500	37.7500	37.75	599.0	0.0	1.0	0.057176	0.057552	0.056800	0.056800	116733.12
14147	1962-02-05	38.000	38.3800	38.0000	38.00	999.0	0.0	1.0	0.056800	0.057368	0.056800	0.056800	194685.12
14148	1962-02-02	38.000	38.5000	38.0000	38.00	1598.0	0.0	1.0	0.056800	0.057547	0.056800	0.056800	311418.24
14149	1962-02-01	37.500	38.6300	37.5000	38.00	3197.0	0.0	1.0	0.056052	0.057741	0.056052	0.056800	623031.36
14150	1962-01-31	37.500	37.6300	37.2500	37.50	799.0	0.0	1.0	0.056800	0.056997	0.056421	0.056800	155709.12
14151	1962-01-30	37.250	37.6300	37.0000	37.50	1698.0	0.0	1.0	0.056421	0.056997	0.056042	0.056800	330906.24
14152	1962-01-29	37.250	37.7500	37.2500	37.25	599.0	0.0	1.0	0.056800	0.057562	0.056800	0.056800	116733.12
14153	1962-01-26	37.500	38.0000	37.2500	37.25	899.0	0.0	1.0	0.057181	0.057943	0.056800	0.056800	175197.12
14154	1962-01-25	38.000	38.1200	37.5000	37.50	399.0	0.0	1.0	0.057557	0.057739	0.056800	0.056800	77757.12
14155	1962-01-24	38.500	38.5000	37.5000	38.00	899.0	0.0	1.0	0.057547	0.057547	0.056052	0.056800	175197.12
14156	1962-01-23	38.630	39.5000	38.2500	38.50	2997.0	0.0	1.0	0.056992	0.058275	0.056431	0.056800	584055.36
14157	1962-01-22	38.250	38.6300	38.2500	38.63	899.0	0.0	1.0	0.056241	0.056800	0.056241	0.056800	175197.12
14158	1962-01-19	37.500	38.0000	37.5000	38.00	1099.0	0.0	1.0	0.056052	0.056800	0.056052	0.056800	214173.12
14159	1962-01-18	36.750	37.5000	36.7500	37.25	799.0	0.0	1.0	0.056037	0.057181	0.056037	0.056800	155709.12
14160	1962-01-17	37.750	37.7500	36.5000	36.50	999.0	0.0	1.0	0.058745	0.058745	0.056800	0.056800	194685.12
14161	1962-01-16	38.500	38.5000	37.5000	37.75	899.0	0.0	1.0	0.057928	0.057928	0.056424	0.056800	175197.12
14162	1962-01-15	38.750	39.0000	38.5000	38.75	1598.0	0.0	1.0	0.066266	0.066694	0.065839	0.066266	311418.24
14163	1962-01-12	40.000	40.1300	38.2500	38.75	5095.0	0.0	1.0	0.068404	0.068626	0.065411	0.066266	992913.60
14164	1962-01-11	38.870	40.2500	38.7500	40.00	5095.0	0.0	1.0	0.064394	0.066680	0.064196	0.066266	992913.60
14165	1962-01-10	38.500	39.1300	38.5000	38.87	1698.0	0.0	1.0	0.065636	0.066710	0.065636	0.066266	330906.24
14166	1962-01-09	37.750	38.5000	37.5000	38.50	1598.0	0.0	1.0	0.055693	0.056800	0.055324	0.056800	311418.24
14167	1962-01-08	37.880	38.3800	37.0000	37.75	3197.0	0.0	1.0	0.056995	0.057748	0.055671	0.056800	623031.36
14168	1962-01-05	37.750	38.0000	37.6300	37.88	2397.0	0.0	1.0	0.056605	0.056980	0.056425	0.056800	467127.36
14169	1962-01-04	37.750	37.8800	37.5000	37.75	2397.0	0.0	1.0	0.056800	0.056995	0.056424	0.056800	467127.36
14170	1962-01-03	37.250	37.8800	37.2500	37.75	1998.0	0.0	1.0	0.056047	0.056995	0.056047	0.056800	389370.24
14171	1962-01-02	37.250	38.5000	37.2500	37.25	2098.0	0.0	1.0	0.056800	0.058706	0.056800	0.056800	408858.24
14172 rows × 13 columns

!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
depot = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
DEP = pd.read_csv(depot, delimiter=',', engine = 'python')
DEPX = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DEPT = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
DEPX = np.array(DEPX)
DEPT = (np.array(DEPT)).reshape(DEPT.shape[0],-1)
--2018-05-01 10:08:36--  https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
Resolving www.quandl.com (www.quandl.com)... 2400:cb00:2048:1::6819:3567, 2400:cb00:2048:1::6819:3667, 104.25.53.103, ...
Connecting to www.quandl.com (www.quandl.com)|2400:cb00:2048:1::6819:3567|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/csv]
Saving to: ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd.1’

HD.csv?api_key=q74T     [ <=>                ]   1.22M  7.26MB/s    in 0.2s    

2018-05-01 10:08:40 (7.26 MB/s) - ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd.1’ saved [1280198]

DEPT
array([[  1.79850000e+02],
       [  1.74910000e+02],
       [  1.74430000e+02],
       ..., 
       [  1.83196935e-02],
       [  1.83196935e-02],
       [  1.83196935e-02]])
!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)
--2018-05-01 10:09:46--  https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
Resolving www.quandl.com (www.quandl.com)... 2400:cb00:2048:1::6819:3567, 2400:cb00:2048:1::6819:3667, 104.25.53.103, ...
Connecting to www.quandl.com (www.quandl.com)|2400:cb00:2048:1::6819:3567|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/csv]
Saving to: ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv’

AAPL.csv?api_key=xz     [ <=>                ]   1.25M  6.49MB/s    in 0.2s    

2018-05-01 10:09:50 (6.49 MB/s) - ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv’ saved [1311730]

APPX
array([[  1.75000000e+02,   1.80250000e+02,   1.74990000e+02],
       [  1.75760000e+02,   1.76240000e+02,   1.74100000e+02],
       [  1.74230000e+02,   1.75600000e+02,   1.73320000e+02],
       ..., 
       [  1.83196935e-02,   1.83196935e-02,   1.83196935e-02],
       [  1.83196935e-02,   1.83196935e-02,   1.83196935e-02],
       [  1.83196935e-02,   1.83196935e-02,   1.83196935e-02]])
APPXTrain, APPTTrain, APPXtest, APPTtest = partition(APPX, APPT, 0.8)
APPXTrain
array([[ 175.        ,  180.25      ,  174.99      ],
       [ 175.76      ,  176.24      ,  174.1       ],
       [ 174.23      ,  175.6       ,  173.32      ],
       ..., 
       [   0.65666342,    0.66213562,    0.65666342],
       [   0.66498116,    0.66760782,    0.65403677],
       [   0.67972119,    0.67972119,    0.66601624]])
APPTTrain
array([[ 162.32      ],
       [ 164.22      ],
       [ 163.65      ],
       ..., 
       [   1.20951306],
       [   1.1947629 ],
       [   1.17263766]])
!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)
APPXTrain, APPTTrain, APPXtest, APPTtest = partition(APPX, APPT, 0.8)
--2018-05-03 19:22:39--  https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
Resolving www.quandl.com (www.quandl.com)... 104.25.53.103, 104.25.54.103
Connecting to www.quandl.com (www.quandl.com)|104.25.53.103|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/csv]
Saving to: ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv.1’

AAPL.csv?api_key=xz     [   <=>              ]   1.25M  3.09MB/s    in 0.4s    

2018-05-03 19:22:43 (3.09 MB/s) - ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv.1’ saved [1312023]

APP
Date	Open	High	Low	Close	Volume	Dividend	Split	Adj_Open	Adj_High	Adj_Low	Adj_Close	Adj_Volume
0	2018-04-30	162.1302	167.2600	161.8400	165.26	42427424.0	0.0	1.0	162.130200	167.260000	161.840000	165.260000	42427424.0
1	2018-04-27	164.0000	164.3300	160.6300	162.32	35655839.0	0.0	1.0	164.000000	164.330000	160.630000	162.320000	35655839.0
2	2018-04-26	164.1200	165.7300	163.3700	164.22	27963014.0	0.0	1.0	164.120000	165.730000	163.370000	164.220000	27963014.0
3	2018-04-25	162.6200	165.4200	162.4100	163.65	28382084.0	0.0	1.0	162.620000	165.420000	162.410000	163.650000	28382084.0
4	2018-04-24	165.6700	166.3300	161.2201	162.94	33692017.0	0.0	1.0	165.670000	166.330000	161.220100	162.940000	33692017.0
5	2018-04-23	166.8348	166.9200	164.0900	165.24	36515477.0	0.0	1.0	166.834800	166.920000	164.090000	165.240000	36515477.0
6	2018-04-20	170.5950	171.2184	165.4300	165.72	65491140.0	0.0	1.0	170.595000	171.218400	165.430000	165.720000	65491140.0
7	2018-04-19	174.9500	175.3900	172.6600	172.80	34808800.0	0.0	1.0	174.950000	175.390000	172.660000	172.800000	34808800.0
8	2018-04-18	177.8100	178.8200	176.8800	177.84	20754538.0	0.0	1.0	177.810000	178.820000	176.880000	177.840000	20754538.0
9	2018-04-17	176.4900	178.9365	176.4100	178.24	26605442.0	0.0	1.0	176.490000	178.936500	176.410000	178.240000	26605442.0
10	2018-04-16	175.0301	176.1900	174.8301	175.82	21578420.0	0.0	1.0	175.030100	176.190000	174.830100	175.820000	21578420.0
11	2018-04-13	174.7800	175.8400	173.8500	174.73	25124255.0	0.0	1.0	174.780000	175.840000	173.850000	174.730000	25124255.0
12	2018-04-12	173.4100	175.0000	173.0400	174.14	22889285.0	0.0	1.0	173.410000	175.000000	173.040000	174.140000	22889285.0
13	2018-04-11	172.2300	173.9232	171.7000	172.44	22431640.0	0.0	1.0	172.230000	173.923200	171.700000	172.440000	22431640.0
14	2018-04-10	173.0000	174.0000	171.5300	173.25	28614241.0	0.0	1.0	173.000000	174.000000	171.530000	173.250000	28614241.0
15	2018-04-09	169.8800	173.0900	169.8450	170.05	29017718.0	0.0	1.0	169.880000	173.090000	169.845000	170.050000	29017718.0
16	2018-04-06	170.9700	172.4800	168.2000	168.38	35005290.0	0.0	1.0	170.970000	172.480000	168.200000	168.380000	35005290.0
17	2018-04-05	172.5800	174.2304	172.0800	172.80	26933197.0	0.0	1.0	172.580000	174.230400	172.080000	172.800000	26933197.0
18	2018-04-04	164.8800	172.0100	164.7700	171.61	34605489.0	0.0	1.0	164.880000	172.010000	164.770000	171.610000	34605489.0
19	2018-04-03	167.6400	168.7455	164.8800	168.39	30278046.0	0.0	1.0	167.640000	168.745500	164.880000	168.390000	30278046.0
20	2018-04-02	167.8800	168.9400	164.4700	166.68	37586791.0	0.0	1.0	167.880000	168.940000	164.470000	166.680000	37586791.0
21	2018-03-29	167.8050	171.7500	166.9000	167.78	38398505.0	0.0	1.0	167.805000	171.750000	166.900000	167.780000	38398505.0
22	2018-03-28	167.2500	170.0200	165.1900	166.48	41668545.0	0.0	1.0	167.250000	170.020000	165.190000	166.480000	41668545.0
23	2018-03-27	173.6800	175.1500	166.9200	168.34	40922579.0	0.0	1.0	173.680000	175.150000	166.920000	168.340000	40922579.0
24	2018-03-26	168.0700	173.1000	166.4400	172.77	37541236.0	0.0	1.0	168.070000	173.100000	166.440000	172.770000	37541236.0
25	2018-03-23	168.3900	169.9200	164.9400	164.94	41028784.0	0.0	1.0	168.390000	169.920000	164.940000	164.940000	41028784.0
26	2018-03-22	170.0000	172.6800	168.6000	168.85	41490767.0	0.0	1.0	170.000000	172.680000	168.600000	168.850000	41490767.0
27	2018-03-21	175.0400	175.0900	171.2600	171.27	37054935.0	0.0	1.0	175.040000	175.090000	171.260000	171.270000	37054935.0
28	2018-03-20	175.2400	176.8000	174.9400	175.24	19649350.0	0.0	1.0	175.240000	176.800000	174.940000	175.240000	19649350.0
29	2018-03-19	177.3200	177.4700	173.6600	175.30	33446771.0	0.0	1.0	177.320000	177.470000	173.660000	175.300000	33446771.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...
9395	1981-01-26	32.3700	32.3700	32.2500	32.25	110000.0	0.0	1.0	0.472318	0.472318	0.470567	0.470567	6160000.0
9396	1981-01-23	32.8700	33.0000	32.7500	32.75	50100.0	0.0	1.0	0.479613	0.481510	0.477863	0.477863	2805600.0
9397	1981-01-22	32.8700	33.1300	32.8700	32.87	158700.0	0.0	1.0	0.479613	0.483407	0.479613	0.479613	8887200.0
9398	1981-01-21	32.5000	32.7500	32.5000	32.50	71000.0	0.0	1.0	0.474215	0.477863	0.474215	0.474215	3976000.0
9399	1981-01-20	32.0000	32.0000	31.8800	31.88	134300.0	0.0	1.0	0.466919	0.466919	0.465168	0.465168	7520800.0
9400	1981-01-19	32.8700	33.0000	32.8700	32.87	185600.0	0.0	1.0	0.479613	0.481510	0.479613	0.479613	10393600.0
9401	1981-01-16	31.1200	31.1200	31.0000	31.00	59800.0	0.0	1.0	0.454079	0.454079	0.452328	0.452328	3348800.0
9402	1981-01-15	31.2500	31.5000	31.2500	31.25	62800.0	0.0	1.0	0.455976	0.459624	0.455976	0.455976	3516800.0
9403	1981-01-14	30.6300	30.7500	30.6300	30.63	63800.0	0.0	1.0	0.446929	0.448680	0.446929	0.446929	3572800.0
9404	1981-01-13	30.6300	30.6300	30.5000	30.50	102900.0	0.0	1.0	0.446929	0.446929	0.445032	0.445032	5762400.0
9405	1981-01-12	31.8800	31.8800	31.6200	31.62	105800.0	0.0	1.0	0.465168	0.465168	0.461374	0.461374	5924800.0
9406	1981-01-09	31.8800	32.0000	31.8800	31.88	96000.0	0.0	1.0	0.465168	0.466919	0.465168	0.465168	5376000.0
9407	1981-01-08	30.3700	30.3700	30.2500	30.25	177800.0	0.0	1.0	0.443135	0.443135	0.441384	0.441384	9956800.0
9408	1981-01-07	31.0000	31.0000	30.8800	30.88	248600.0	0.0	1.0	0.452328	0.452328	0.450577	0.450577	13921600.0
9409	1981-01-06	32.3700	32.3700	32.2500	32.25	201600.0	0.0	1.0	0.472318	0.472318	0.470567	0.470567	11289600.0
9410	1981-01-05	33.8700	33.8700	33.7500	33.75	159500.0	0.0	1.0	0.494205	0.494205	0.492454	0.492454	8932000.0
9411	1981-01-02	34.5000	34.7500	34.5000	34.50	96700.0	0.0	1.0	0.503397	0.507045	0.503397	0.503397	5415200.0
9412	1980-12-31	34.2500	34.2500	34.1300	34.13	159600.0	0.0	1.0	0.499749	0.499749	0.497998	0.497998	8937600.0
9413	1980-12-30	35.2500	35.2500	35.1200	35.12	307500.0	0.0	1.0	0.514341	0.514341	0.512444	0.512444	17220000.0
9414	1980-12-29	36.0000	36.1300	36.0000	36.00	415900.0	0.0	1.0	0.525284	0.527181	0.525284	0.525284	23290400.0
9415	1980-12-26	35.5000	35.6200	35.5000	35.50	248100.0	0.0	1.0	0.517988	0.519739	0.517988	0.517988	13893600.0
9416	1980-12-24	32.5000	32.6300	32.5000	32.50	214300.0	0.0	1.0	0.474215	0.476112	0.474215	0.474215	12000800.0
9417	1980-12-23	30.8800	31.0000	30.8800	30.88	209600.0	0.0	1.0	0.450577	0.452328	0.450577	0.450577	11737600.0
9418	1980-12-22	29.6300	29.7500	29.6300	29.63	166800.0	0.0	1.0	0.432338	0.434089	0.432338	0.432338	9340800.0
9419	1980-12-19	28.2500	28.3800	28.2500	28.25	217100.0	0.0	1.0	0.412202	0.414099	0.412202	0.412202	12157600.0
9420	1980-12-18	26.6300	26.7500	26.6300	26.63	327900.0	0.0	1.0	0.388564	0.390315	0.388564	0.388564	18362400.0
9421	1980-12-17	25.8700	26.0000	25.8700	25.87	385900.0	0.0	1.0	0.377475	0.379372	0.377475	0.377475	21610400.0
9422	1980-12-16	25.3700	25.3700	25.2500	25.25	472000.0	0.0	1.0	0.370179	0.370179	0.368428	0.368428	26432000.0
9423	1980-12-15	27.3800	27.3800	27.2500	27.25	785200.0	0.0	1.0	0.399508	0.399508	0.397611	0.397611	43971200.0
9424	1980-12-12	28.7500	28.8700	28.7500	28.75	2093900.0	0.0	1.0	0.419498	0.421249	0.419498	0.419498	117258400.0
9425 rows × 13 columns

The higher the number of iterations with the classifier, the more accurate the result

def rollingWindows(X, windowSize):
    nSamples, nAttributes = X.shape
    nWindows = nSamples - windowSize + 1
    # Shape of resulting matrix
    newShape = (nWindows, nAttributes * windowSize)
    itemSize = X.itemsize  # number of bytes
    # Number of bytes to increment to starting element in each dimension
    strides = (nAttributes * itemSize, itemSize)
    return np.lib.stride_tricks.as_strided(X, shape=newShape, strides=strides)
APPXw = rollingWindows(APPX, 10)
APPXw
array([[  1.75000000e+02,   1.75760000e+02,   1.74230000e+02, ...,
          1.79000000e+02,   1.79440000e+02,   1.78820000e+02],
       [  1.74490000e+02,   1.73310000e+02,   1.72000000e+02, ...,
          1.81701993e+02,   1.76840036e+02,   1.76700838e+02],
       [  1.74030000e+02,   1.75060000e+02,   1.77980000e+02, ...,
          1.81701993e+02,   1.83352474e+02,   1.87786897e+02],
       ..., 
       [  2.74795403e-02,   2.74795403e-02,   3.66393870e-02, ...,
          1.83196935e-02,   1.83196935e-02,   1.83196935e-02],
       [  3.66393870e-02,   3.66393870e-02,   3.66393870e-02, ...,
          1.83196935e-02,   1.83196935e-02,   1.83196935e-02],
       [  3.66393870e-02,   2.74795403e-02,   2.74795403e-02, ...,
          1.83196935e-02,   1.83196935e-02,   1.83196935e-02]])
APPTw = rollingWindows(APPT, 10)
APPTw
array([[ 162.32      ,  164.22      ,  163.65      , ...,  177.84      ,
         178.24      ,  175.82      ],
       [ 164.22      ,  163.65      ,  162.94      , ...,  178.24      ,
         175.82      ,  174.73      ],
       [ 163.65      ,  162.94      ,  165.24      , ...,  175.82      ,
         174.73      ,  174.14      ],
       ..., 
       [   0.51244373,    0.525284  ,    0.51798839, ...,    0.38856425,
           0.37747492,    0.36842836],
       [   0.525284  ,    0.51798839,    0.47421473, ...,    0.37747492,
           0.36842836,    0.39761081],
       [   0.51798839,    0.47421473,    0.45057695, ...,    0.36842836,
           0.39761081,    0.41949764]])
keep = [len(np.unique(Trow))==1 for Trow in APPTw]
sum(keep)
0
