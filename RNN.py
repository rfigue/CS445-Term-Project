import time
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


def plotResults(predicted, actual, length):
    fig = plt.figure(facecolor='white')
    fig2 = fig.add_subplot(111)
    fig2.plot(actual, label='Actual')
    print('T0')

    for i, data in enumerate(predicted):
        padding = [None for p in xrange(i * length)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def normalize(nWindows):
    n = []
    for i in nWindows:
        n = [((float(p) / float(i[0])) - 1) for p in i]
        n.append(n)
    return n

def model(layers):
    model = Sequential()
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def onePointPrediction(model, data):
    predicted = model.predict(data)
    return np.reshape(predicted, (predicted.size,))

def newPointPrediction(model, data, window):
    current = data[0]
    predicted = []
    for i in xrange(len(data)):
        predicted.append(model.predict(current[newaxis,:,:])[0,0])
        current = current[1:]
        current = np.insert(current, [window-1], predicted[-1], axis=0)
    return predicted

def multiPointPredictions(model, data, window, length):
    predicted = []
    for i in xrange(len(data)/length):
        current = data[i*length]
        pred = []
        for item in xrange(length):
            pred.append(model.predict(current[newaxis,:,:])[0,0])
            current = current[1:]
            current = np.insert(current, [window-1], pred[-1], axis=0)
        predicted.append(pred)
    return predicted
