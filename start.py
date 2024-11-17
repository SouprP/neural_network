import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import matplotlib.pyplot as plt
 
import argparse

from core.neural_network import NeuralNetwork

# from keras.api.datasets import mnist

def test_prediction(index, W1, b1, W2, b2, x_train, y_train, network):
    current_image = x_train[:, index, None]
    prediction = network.make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    # use args in the future for choosing diffrent things, example - activation functions
    # sigmoid / relu / other 

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--arg1')
    # args = parser.parse_args()

    # print(args.arg1)
    (x_train, y_train), (x_test, y_test) = keras.api.datasets.mnist.load_data()

    # assert x_train.shape == (60000, 28, 28) # the images 28x28 pixels
    # assert x_test.shape == (10000, 28, 28)
    # assert y_train.shape == (60000,) # output label - 1, 2, 3, ..., 9
    # assert y_test.shape == (10000,)

    x_train = x_train.reshape(60000, 784).T / 255
    x_test = x_test.reshape(10000, 784).T / 255
    y_train = y_train.T
    y_test = y_test.T

    network = NeuralNetwork()
    W1, b1, W2, b2 = network.gradient_descent(x_train, y_train, 0.10, 500)

    test_prediction(np.random.randint(0, 60000) , W1, b1, W2, b2, x_train, y_train, network)
    test_prediction(np.random.randint(0, 60000), W1, b1, W2, b2, x_train, y_train, network)
    test_prediction(np.random.randint(0, 60000), W1, b1, W2, b2, x_train, y_train, network)
    test_prediction(np.random.randint(0, 60000), W1, b1, W2, b2, x_train, y_train, network)

