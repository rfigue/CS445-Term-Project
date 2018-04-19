import pandas as pd
import numpy as np
import mlutilities as ml

!wget https://www.quandl.com/api/v3/datasets/EOD/DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd
#Disney stock
fName = 'DIS.csv?api_key=q74Toz9W7ovpxJh3Vybd'
disney = pd.read_csv(fName, delimiter=',', engine = 'python')
