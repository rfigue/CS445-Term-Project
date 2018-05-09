import time
import numpy as np
from numpy import newaxis
#The code requires installation of keras, which is a dependency of this class
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


def plotResults(predicted, actual, length):
    fig = plt.figure(facecolor='white')
    fig2 = fig.add_subplot(111)
    fig2.plot(actual, label='Actual Close')
    print('T0')

    for i, item in enumerate(predicted):
        for p in xrange(i * length):
            temp = [None]
        plt.plot(item + temp, label='Predicted Close')
        plt.legend()
    plt.show()

def normalize(nWindows):
    resNorm = []
    for w in nWindows:
        for i in w:
            norm = [(float(i) / float(w[0])) - 1]
        resNorm.append(norm)
    return resNorm

def model(layers):
    model = Sequential()
    model.add(LSTM(input_dim = layers[0], output_dim = layers[1], return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss = "rmse", optimizer = "rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def firstPoint(model, data):
    resPredictions = model.predict(data)
    return np.reshape(resPredictions, (resPredictions.size,))

def newPoint(model, data, window):
    predicted = []
    current = data[0]
    for i in xrange(len(data)):
        predicted.append(model.predict(current[newaxis,:,:])[0,0])
        current = current[1:]
        current = np.insert(current, [window-1], predicted[-1], axis=0)
    return predicted

def multiPoint(model, data, window, length):
    predicted = []
    for i in xrange(len(data)/length):
        current = data[i*length]
        tempPreds = []
        for item in xrange(length):
            tempPreds.append(model.predict(current[newaxis,:,:])[0,0])
            current = current[1:]
            current = np.insert(current, [window-1], tempPreds[-1], axis=0)
        predicted.append(tempPreds)
    return predicted
