import time
start = time.time()

import pandas as pd
import numpy as np

def initializeData(file):
    # reading in data using pandas; shape = 59999, 785
    Dataframe = pd.read_csv(file)

    # converting dataframe to numpy array for easier manipulation
    data = np.array(Dataframe)
    np.random.shuffle(data) # shuffling for different results each time

    # separating first 1000 images to be used to test the model's accuracy
    testArray = data[0:1000].T
    # using the other 58000 to train the model
    trainingArray = data[1000:,].T

    # the first column is what each of the 60000 drawn numbers are
    Y_test = testArray[0]
    Y_train = trainingArray[0]

    # the other columns are individual pixel values. dividing by 255 because data between 0 and 1 is easier to manage
    X_test = testArray[1:,] / 255   
    X_train = trainingArray[1:,] / 255

    return X_test, Y_test, X_train, Y_train

def initializeLearningParameters(layerSizesList):
    params = dict()
    # generating random weights and biases to be adjusted over time; learning must start with an attempt
    for i in range(1, len(layerSizesList)):
        params[f'W{i}'] = np.random.rand(layerSizesList[i], layerSizesList[i-1]) - 0.5
        params[f'b{i}'] = np.random.rand(layerSizesList[i], 1) - 0.5
    return params

# activation functions to add non-linearity; with only linear modifications layers have no effect
def reLu(z):
    return np.maximum(z, 0)

def sigmoid(x):    
    return 1/(1 + np.exp(-(x)))

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# # derivatives of activation functions to be used to back-track. 
def reLu_deriv(z):
    return z > 0

def sigmoidDeriv(z):
    y = sigmoid(z)
    return y * (1 - y)

def comeUpWithAnAnswer(params, X, layerSizesList, activations):
    connectionStorage = dict()
    for i in range(1, len(layerSizesList)):
        W = params[f'W{i}']
        b = params[f'b{i}']

        # the most recent inputs to be considered are the last layer's activation values or the original input if no previous layer exists.
        neuronValues = connectionStorage.get(f'A{i-1}', X)
        
        connectionStorage[f'Z{i}'] = W.dot(neuronValues) + b

        # little loss of efficiency vs if statements but cleaner
        activationDict = { 
            'sigmoid': sigmoid(connectionStorage[f'Z{i}']),
            'reLu': reLu(connectionStorage[f'Z{i}']),
            'softmax': softmax(connectionStorage[f'Z{i}'])
        }

        connectionStorage[f'A{i}'] = activationDict[activations[i-1]]   # the activation list starts at 0 but the cS storage count starts at 1

    return connectionStorage

# 1/11 3:14 diff because whole matrix
# to format a single integer into a subtract-able array; example of difficulty in math application
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# 1/15, loss is the loss of a single set whereas Cost is the average loss of the whole. Cost is what is minimized
def reflect(params, connectionStorage, X, Y, numLayers, activations):
    partials = dict()
    A_final = connectionStorage[f'A{numLayers}']

    # derivative of cross-entropy loss function times the derivative of softmax; used bc found to be the preeminent loss-function/final activation combination
    dZ_final = A_final - one_hot(Y)
        
    m = X.shape[1]  # number of images

    # multiplying previous A because thats dZ2/dW2 and then dividing by m because dot product something or other (still got to figure that out). Transposed for organization and matching of dimensions
    dW_final = dZ_final.dot(connectionStorage[f'A{numLayers-1}'].T) * (1 / m)

    # the average of how much the output was off by
    # 1/10 Because the amount that the last bias affected the loss (and should be changed) is ideally just the amount lossed; this will approach 0 as accuracy improves 
    db_final = np.sum(dZ_final) * (1 / m)  

    # storing the adjustments
    partials[f'dZ{numLayers}'] = dZ_final
    partials[f'dW{numLayers}'] = dW_final
    partials[f'db{numLayers}'] = db_final

    # back-tracking through layers
    for i in range(numLayers-1, 0, -1):

        if activations[i-1] == 'reLu':
            currentX = connectionStorage.get(f'A{i-1}', X)

            # 1/29 1:58 many magical .Ts, prove
            partials[f'dZ{i}'] = params[f'W{i+1}'].T.dot(partials[f'dZ{i+1}']) * reLu_deriv(connectionStorage[f'Z{i}'])

            # same logic as for dW_final
            partials[f'dW{i}'] = partials[f'dZ{i}'].dot(currentX.T) * (1 / m)
            # same as for db_final, but with respect to the current dZ. not sure here
            partials[f'db{i}'] = np.sum(partials[f'dZ{i}']) * (1 / m)

    return partials

def correctParams(params, partials, numLayers, learningRate):
    
    for i in range(1, numLayers):
        params[f'W{i}'] = params[f'W{i}'] - partials[f'dW{i}'] * learningRate
        params[f'b{i}'] = params[f'b{i}'] - partials[f'db{i}'] * learningRate

    return params

def get_predictions(A):
    # returns the index of the max value in the output (size 10)
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    return f'{round((np.sum(predictions == Y) / Y.size) * 100, 2)}%'

def gradient_descent(X, Y, layerSizesList, activations, learningRate, numTrials):

    params = initializeLearningParameters(layerSizesList)
    for i in range(numTrials+1):
        connectionStorage = comeUpWithAnAnswer(params, X, layerSizesList, activations)

        partials = reflect(params, connectionStorage, X, Y, len(layerSizesList)-1, activations)
        
        params = correctParams(params, partials, len(layerSizesList), learningRate)
        
        if i % 10 == 0:
            A_final = connectionStorage[f'A{len(activations)}']
            print("Iteration: ", i)
            predictions = get_predictions(A_final)
            print(get_accuracy(predictions, Y))

    return params

from matplotlib import pyplot as plt
def showNumber(pixels):
    plt.gray()
    plt.imshow(pixels, interpolation='nearest')
    plt.show()

def main():
    # splitting input/correction data into training and testing sets
    # X = input, Y = desired output
    X_test, Y_test, X_train, Y_train = initializeData('/Users/nlehr24/Downloads/mnist_train.csv')    # downloaded from https://pjreddie.com/projects/mnist-in-csv/#google_vignette. Only loading in training data for personal splitting.

    layerSizesList = [X_test.shape[0], 32, len(np.unique(Y_test))]  # using test data for shape just so less processing

    activations = ['reLu', 'softmax']   # preeminent activations 

    final_params = gradient_descent(X_train, Y_train, layerSizesList, activations, 0.1, 200)


    A_final = comeUpWithAnAnswer(final_params, X_test, layerSizesList, activations)[f'A{len(layerSizesList) - 1}']
    print(f'Accuracy on test data: {round((np.sum(np.argmax(A_final, axis=0) == Y_test) / Y_test.size) * 100, 2)}%')
    # 85% on test

    numImagesShown = 4
    
    for i in range(numImagesShown):
        print(f'Image {i+1} guess: {np.argmax(A_final[:, i], axis=0)}')

    for i in range(numImagesShown):
        showNumber(X_test[:, i].reshape((28, 28)) * 255)


main()

print('time taken:', round(time.time() - start, 2))
