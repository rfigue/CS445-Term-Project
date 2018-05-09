
# CS 445: Using Time Embedding to Forecast Stock Behavior

#### Team Members: Rodolfo Figueroa, Evan Salzman

## Introduction

In this report we will discuss the process of forecasting stock prices by referencing their past behavior.  In order to accomplish this, the team has implemented code to use time embedding to record the changes of six highlighted stocks over many years.  The stocks that will be used throughout this investigation include: Disney (DIS), Apple (AAPL), Home Depot (HD), Nike (NKE), Intel (INTC) and Microsoft (MSFT) from the NYSE (New York Stock Exchange).  The CSV files containing the data from these stocks were found at: https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices.

## Description of Datasets

The data sets for the six stocks chosen for this project are DIS, AAPL, HD, NKE, INTC and MSFT from the NYSE.  Each of these files is downloaded in this notebook using pandas from the Quandl site providing End of Day US Stock Prices.  Each .csv file contains the following attributes:

* Date	*(1.2.1962 - 4.19.2018)*
* Open	*(decimal value of the price of the stock when it opened on the given day)*
* High	*(decimal value the peak in price of the stock for the given day)*
* Low	*(decimal value the lowest price of the stock for the given day)*
* Close	*(decimal value the final price of the stock from the market closed on that the given day)*
* Volume	*(a value to show the amount of traffic the stock experience on the given day)*
* Dividend	*(a value to show the unadjusted dividend on any ex-dividend if applicable, else 0.0.)*
* Split	    *(a value to show any split that occurred on the given DATE, else 1.0)*
* Adj_Open	*(adjusted decimal value of the price of the stock when it opened on the given day)*
* Adj_High	*(adjusted decimal value of the peak in price of the stock for the given day)*
* Adj_Low	*(adjusted decimal value of the lowest price of the stock for the given day)*
* Adj_Close	*(adjusted decimal value of the final price of the stock from when the market closed on the given day)*
* Adj_Volume *(an adjusted value to show the amount of traffic the stock experience on the given day)*

The values that have been adjusted now reflect any dividends and splits that occurred on each specific day.  The values that are adjusted are done with these respective calculations detailed at the following link (http://www.crsp.com/products/documentation/crsp-calculations): 

    Price and dividend data are adjusted with the calculation:

        A(t) = P(t) / C(t),

        where A(t) is the adjusted value at time t, P(t) is the raw value at time t, and C(t) is the cumulative 
        adjustment factor at time t.

    Share and volume data are adjusted with the calculation:

        A(t) = P(t) * C(t),

        where A(t) is the adjusted value at time t, P(t) is the raw value at time t, and C(t) is the cumulative 
        adjustment factor at time t.

For the team's purposes in this assignment, the adjusted values will be used as they are more representative of how the stocks are performing in the marketing.  From these "Adj" values, the Open, High and Low will be the input values, stored in the X matrix, while the Close will be the one and only target value in the T matrix.  The stock price at the end of the day is the most important to consider as it is a reflection of the entire day, as it relates to how well the stock is performing compared to other days in its past and in comparison to other stocks.  Through using these datasets and setting up the data in this way. it is possible to predict how each respective stock will perform in the market. 

## Introduction to Recurrent Neural Networks

The term recurrent comes from the idea that a task is done over and over again on every element in a sequence, with the result dependent on all of the previous computations done with the network.  Additionally, recurrent neural network is able to memorize and store information computed in previous calculations using the network, which makes them ideal for this task of forecasting stock prices on the market.  One of the components of a RNN that is most helpful in this process is the hidden state, which it maintains to store information about the sequence it processes.  In this task, the team will be making using of this hidden state to keep data that applies to the stock market data used as input. 

## Dependencies

The Recurrent Neural Network (RNN.py) class has the following dependency:
* Keras (and the specific components below)
    * Dense, Activation, Dropout
    * LSTM
    * Sequential

Therefore, in order for the RNN class to function, the keras external library must be downloaded and made available in the local directory containing this notebook, the RNN.py file and the mlutilities.py file.  Information/documnetation on keras is available here: https://keras.io/.  Keras worked well as a package with deep learning tools.


## Downloading, Organizing and Partitioning Stock Data and Prices

The code below downloads the six chosen stocks (DIS, AAPL, HD, NKE, INTC and MSFT) from the Quandl site publishing end of day data for each of them.  The code cell reads the csv input files with pandas and makes use of the mlutilities.py file built upon throughout the semester in CS445.  This file contains a partition function that will be used to partition the data, which has been read in with pandas into numpy arrays, and place it into sets of training and testing data for the data in the X and T matrices.


```python
import pandas as pd
import numpy as np
import mlutilities as ml
#import neuralnetworksA4 as nn
```

The code below downloads the Disney stock (DIS) from the Quandl site publishing end of day data for this stock beginning in 1962 until the time it was downloaded for this assignment.  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Disney stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
```

    --2018-05-01 10:05:19--  https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
    Resolving www.quandl.com (www.quandl.com)... 2400:cb00:2048:1::6819:3567, 2400:cb00:2048:1::6819:3667, 104.25.53.103, ...
    Connecting to www.quandl.com (www.quandl.com)|2400:cb00:2048:1::6819:3567|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd.4’
    
    DIS.csv?api_key=q74     [  <=>               ]   1.87M  8.81MB/s    in 0.2s    
    
    2018-05-01 10:05:25 (8.81 MB/s) - ‘DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd.4’ saved [1965115]
    



```python
disney = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
```


```python
DIS = pd.read_csv(disney, delimiter=',', engine = 'python')
```


```python
DISX = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
DIST = pd.read_csv(disney, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
```


```python
DISX = np.array(DISX)
DIST = (np.array(DIST)).reshape(DIST.shape[0],-1)
```

The code below downloads the Home Depot stock (HD).  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Home Depot stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Home Depot
depot = 'HD.csv?api_key=q74Toz9W7ovpxJh3Vybd'
HD = pd.read_csv(depot, delimiter=',', engine = 'python')
HDX = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
HDT = pd.read_csv(depot, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
HDX = np.array(HDX)
HDT = (np.array(HDT)).reshape(HDT.shape[0],-1)
HDXTrain, HDTTrain, HDXtest, HDTtest = ml.partition(HDX, HDT, 0.8)
```

    --2018-05-08 11:15:22--  https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=q74Toz9W7ovpxJh3Vybd
    Resolving www.quandl.com (www.quandl.com)... 104.25.54.103, 104.25.53.103
    Connecting to www.quandl.com (www.quandl.com)|104.25.54.103|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd.3’
    
    HD.csv?api_key=q74T     [  <=>               ]   1.22M  5.11MB/s    in 0.2s    
    
    2018-05-08 11:15:26 (5.11 MB/s) - ‘HD.csv?api_key=q74Toz9W7ovpxJh3Vybd.3’ saved [1280667]
    


The code below downloads the Apple stock (APP).  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Apple stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Apple
apple = 'AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
APP = pd.read_csv(apple, delimiter=',', engine = 'python')
APPX = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
APPT = pd.read_csv(apple, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
APPX = np.array(DEPX)
APPT = (np.array(APPT)).reshape(APPT.shape[0],-1)
APPXTrain, APPTTrain, APPXtest, APPTtest = ml.partition(APPX, APPT, 0.8)
```

    --2018-05-08 11:12:38--  https://www.quandl.com/api/v3/datasets/EOD/AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv
    Resolving www.quandl.com (www.quandl.com)... 104.25.54.103, 104.25.53.103
    Connecting to www.quandl.com (www.quandl.com)|104.25.54.103|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv.2’
    
    AAPL.csv?api_key=xz     [ <=>                ]   1.25M  7.37MB/s    in 0.2s    
    
    2018-05-08 11:12:43 (7.37 MB/s) - ‘AAPL.csv?api_key=xzVEv6Le8ghyfmj4XXHv.2’ saved [1312217]
    


The code below downloads the Nike stock (NIK).  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Nike stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Nike
nike = 'NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd'
NIK = pd.read_csv(nike, delimiter=',', engine = 'python')
NIKX = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
NIKT = pd.read_csv(nike, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
NIKX = np.array(NIKX)
NIKT = (np.array(NIKT)).reshape(NIKT.shape[0],-1)
NIKXTrain, NIKTTrain, NIKXtest, NIKTtest = ml.partition(NIKX, NIKT, 0.8)
```

    --2018-05-08 11:11:51--  https://www.quandl.com/api/v3/datasets/EOD/NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd
    Resolving www.quandl.com (www.quandl.com)... 104.25.54.103, 104.25.53.103
    Connecting to www.quandl.com (www.quandl.com)|104.25.54.103|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd’
    
    NKE.csv?api_key=q74     [  <=>               ]   1.23M  5.29MB/s    in 0.2s    
    
    2018-05-08 11:11:56 (5.29 MB/s) - ‘NKE.csv?api_key=q74Toz9W7ovpxJh3Vybd’ saved [1290080]
    


The code below downloads the Microsoft stock (MSFT).  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Microsoft stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Microsoft
micro = 'MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
MSFT = pd.read_csv(micro, delimiter=',', engine = 'python')
MSFTX = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
MSFTT = pd.read_csv(micro, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
MSFTX = np.array(MSFTX)
MSFTT = (np.array(MSFTT)).reshape(MSFTT.shape[0],-1)
MSFTTrain, MSFTTTrain, MSFTXtest, MSFTTtest = ml.partition(MSFTX, MSFTT, 0.8)
```

    --2018-05-08 11:17:43--  https://www.quandl.com/api/v3/datasets/EOD/MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv
    Resolving www.quandl.com (www.quandl.com)... 104.25.54.103, 104.25.53.103
    Connecting to www.quandl.com (www.quandl.com)|104.25.54.103|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv.1’
    
    MSFT.csv?api_key=xz     [ <=>                ]   1.07M  5.77MB/s    in 0.2s    
    
    2018-05-08 11:17:46 (5.77 MB/s) - ‘MSFT.csv?api_key=xzVEv6Le8ghyfmj4XXHv.1’ saved [1123160]
    


The code below downloads the Intel stock (INT).  The code cell reads the csv input file with pandas, separates the important attributes into X (input) and T (output) matrices converted into numpy arrays.  Next, the partition file from mlutilities.py is applied to these two matrices to split each of them into training and testing data for the Intel stock.


```python
!wget https://www.quandl.com/api/v3/datasets/EOD/INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv
#Intel
intel = 'INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv'
INT = pd.read_csv(intel, delimiter=',', engine = 'python')
INTX = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(8, 9, 10), engine = 'python')
INTT = pd.read_csv(intel, delimiter=',', skiprows = 1, usecols=(11,), engine = 'python')
INTX = np.array(INTX)
INTT = (np.array(INTT)).reshape(INTT.shape[0],-1)
INTXTrain, INTTTrain, INTXtest, INTTtest = ml.partition(INTX, INTT, 0.8)
```

    --2018-05-08 11:12:55--  https://www.quandl.com/api/v3/datasets/EOD/INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv
    Resolving www.quandl.com (www.quandl.com)... 104.25.54.103, 104.25.53.103
    Connecting to www.quandl.com (www.quandl.com)|104.25.54.103|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [text/csv]
    Saving to: ‘INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv’
    
    INTC.csv?api_key=xz     [  <=>               ]   1.27M  4.23MB/s    in 0.3s    
    
    2018-05-08 11:12:59 (4.23 MB/s) - ‘INTC.csv?api_key=xzVEv6Le8ghyfmj4XXHv’ saved [1333779]
    


## Experimentation with Stock Data and Prices

The higher the number of iterations with the classifier, the more accurate the result


```python
def rollingWindows(X, windowSize):
    nSamples, nAttributes = X.shape
    nWindows = nSamples - windowSize + 1
    # Shape of resulting matrix
    newShape = (nWindows, nAttributes * windowSize)
    itemSize = X.itemsize  # number of bytes
    # Number of bytes to increment to starting element in each dimension
    strides = (nAttributes * itemSize, itemSize)
    return np.lib.stride_tricks.as_strided(X, shape=newShape, strides=strides)

```


```python
APPXw = rollingWindows(APPX, 10)
APPXw
```




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




```python
APPTw = rollingWindows(APPT, 10)
APPTw
```




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




```python
keep = [len(np.unique(Trow))==1 for Trow in APPTw]
```


```python
sum(keep)
```




    0



## Conclusion and Further Applications

This process of forecasting stocks was a useful application of machine learning to something that is relevant to the real world and for that reason, among others, it was a worthwhile experience.  The most difficult part of the planning process was selecting which type of neural network and embedding technique to use to store data about stocks.  The reasonable choices were time embedding and recurrent neural networks, which had both been discussed in the lectures and seemed like good tools for making predictions about future stock behavior based on that of the past.

In the end, the team chose to use a recurrent neural network because of its ability to essentially memorize information about a lineage of data, which in this case comes from EOD stock data from 1962 to 2018.  This memory is necessary to make such predictions that relate to forecasting a stock in the days, weeks, months and years to come.  While the team recognizes that RNNs are not perfect for maintaining all data and past performance permanently or most effectively, of the tools which are available and within the team's comprehension, this was the best option.  The RNN was tricky to implement and was the most difficult part of this project in terms of the code.  Since it is not a type of neural network used during a previous assignment in CS 445, there was also not much experience among the team members when it came to writing a recurrent neural network.  Seemingly, the first couple weeks of this task went into researching these networks, as well as time embedding, to see which one will work better for the team to implement and also how to fully implement one.     

Further effort with this project and the methods used within it would be beneficial in deciding which stocks to invest in and perhaps actually investing in them.  Likewise, this project could be built into a service to assist people in making sound predictions about how a stock will behave in the future so they can make better investment decisions.  Nevertheless, there are factors that are outside the control of the program and the data it has accumulated on each given stock, such sudden economic, social, political and/or societal matters which affect the overall economy or the economy of a specific company.  An example of this could be a scandal with a stock that causes people to stop investing, sell or at a minimum begin to question the company.  This type of situation is not addressed or even made possible for the tools and methods implemented in this assignment.  Seemingly, another direction for this assignment in the case that the team chose to continue with it, may be to find a way, if possible, to acknowledge these factors once they happen and use it to better forecast the potentiality of the stock to do well or poorly in the future of the NYSE.


