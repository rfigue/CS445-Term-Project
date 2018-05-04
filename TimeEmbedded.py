#Time Embedding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import neuralnetworksA4 as nn
import qdalda
import mlutilities as ml

class TimeEmbedded:

    def rollingWindows(X, windowSize):
        nSamples, nAttributes = X.shape
        nWindows = nSamples - windowSize + 1
        # Shape of resulting matrix
        newShape = (nWindows, nAttributes * windowSize)
        itemSize = X.itemsize  # number of bytes
        # Number of bytes to increment to starting element in each dimension
        strides = (nAttributes * itemSize, itemSize)
        return np.lib.stride_tricks.as_strided(X, shape=newShape, strides=strides)
