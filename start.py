import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tkinter as tk

from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
from core.neural_network_ackley import NeuralNetworkAckley
from core.neural_network_mnist import NeuralNetworkMnist
from core.paint_app import PaintApp
from openpyxl import Workbook
import multiprocessing

# PARAMS
INPUT_SIZE_MNIST = 784  # 28x28 pixels
OUTPUT_SIZE_MNIST = 10  # digits 0 to 9
INPUT_SIZE_ACKLEY = 2      # Ackley input
OUTPUT_SIZE_ACKLEY = 1     # Ackley output


HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 64
ALPHA = 0.01
ITERATIONS = 2500 + 1

# TESTING PARAMS
TEST_BATCH = 20
DEBUG = True

HIDDEN_ARRAY_1 = [32, 64, 128, 256]
HIDDEN_ARRAY_2 = [32, 64, 128, 256]
ALPHA_ARRAY = [0.01, 0.05, 0.1]
# HIDDEN_ARRAY_1 = [256]
# HIDDEN_ARRAY_2 = [32, 64, 128, 256]
# ALPHA_ARRAY = [0.1]
# HIDDEN_ARRAY_2 = [32]

RELU = "relu"
SIGMOID = "sigmoid"

# Tworzenie skoroszytu Excel
# workbook = Workbook()
# sheet = workbook.active
# sheet.title = "Results"
# sheet.append(["Iteration", "Training Accuracy", "Test Accuracy", "Prediction", "Actual Label"])

def ackley_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# assert x_train.shape == (60000, 28, 28) # the images 28x28 pixels
# assert x_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,) # output label - 1, 2, 3, ..., 9
# assert y_test.shape == (10000,)

# Reshape and normalize data
x_train = x_train.reshape(60000, 784).T / 255
x_test = x_test.reshape(10000, 784).T / 255
y_train = y_train.T
y_test = y_test.T

# ACKLEY DATA
x_train_ackley = np.random.uniform(-2, 2, (INPUT_SIZE_ACKLEY, 20000))
y_train_ackley = ackley_function(x_train_ackley[0, :], x_train_ackley[1, :]).reshape(1, -1)
x_test_ackley = np.random.uniform(-2, 2, (INPUT_SIZE_ACKLEY, 5000))
y_test_ackley = ackley_function(x_test_ackley[0, :], x_test_ackley[1, :]).reshape(1, -1)


def ackley_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

def test_prediction(x_test, y_test, network: NeuralNetworkMnist):
    # index = np.random.randint(0, x_test.shape[1] - 1)
    index = 102

    image = x_test[:, index, None]
    label = y_test[index]

    prediction = network.predict(image)

    print(f"Prediction: {prediction[0]}")
    print(f"Actual Label: {label}")

    # sheet.append(["", "", "", prediction[0], label])

    # Reshape and display the image (e.g., 28x28 for MNIST)
    image_size = int(x_train.shape[0] ** 0.5)  # Calculate square size, e.g., 28x28
    reshaped_image = image.reshape((image_size, image_size)) * 255  # Scale for display
    plt.gray()
    plt.imshow(reshaped_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]} | Label: {label}")
    plt.show()

if __name__ == "__main__":
    # use args in the future for choosing diffrent things, example - activation functions
    # sigmoid / relu / other

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu', help="Choose activation function: relu or sigmoid")
    args = parser.parse_args()
    activation_function = args.activation



    # network = NeuralNetworkMnist(INPUT_SIZE_MNIST, HIDDEN_SIZE1, HIDDEN_ARRAY_2, OUTPUT_SIZE_MNIST, RELU
    #                              ,"0.1_256_256_relu_mnist" , False, 1)

    network = NeuralNetworkMnist(INPUT_SIZE_MNIST, HIDDEN_SIZE1, HIDDEN_ARRAY_2, OUTPUT_SIZE_MNIST, SIGMOID
                                 ,"0.1_256_256_sigmoid_mnist" , False, 1)

    # test_prediction(x_test, y_test, network)

    root = tk.Tk()
    app = PaintApp(root, network)
    root.mainloop()


    if False:
        for i in range(TEST_BATCH):
            test_prediction(x_test, y_test, network)
