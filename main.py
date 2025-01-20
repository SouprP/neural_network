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
ITERATIONS = 5000 + 1

# TESTING PARAMS
TEST_BATCH = 20
DEBUG = True

# HIDDEN_ARRAY_1 = [32, 64, 128, 256]
# HIDDEN_ARRAY_2 = [32, 64, 128, 256]
# ALPHA_ARRAY = [0.01, 0.05, 0.1]
HIDDEN_ARRAY_1 = [256]
HIDDEN_ARRAY_2 = [32, 64, 128, 256]
ALPHA_ARRAY = [0.1]
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

def process_mnist_sigmoid():
    offset = 15
    iter = 0
    for a in range(len(ALPHA_ARRAY)):
        for x in range(len(HIDDEN_ARRAY_1)):
            for y in range(len(HIDDEN_ARRAY_2)):
                network = NeuralNetworkMnist(
                    INPUT_SIZE_MNIST, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_MNIST,
                    activation_function=SIGMOID,
                    # filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_sigmoid_mnist",
                    filename=None,
                    iters_check=1
                )
                print(f"SIGMOID MNIST Iteration: {iter}")
                network.train_multiple(
                    x_train, y_train, ALPHA_ARRAY[a], ITERATIONS,
                    excel_filename="merge/fix_0_1_mnist_sigmoid_excel.xlsx",
                    start_col=iter * 4 + 1 + offset
                )
                network.save_data()
                iter += 1

def process_relu_ackley():
    offset = 15
    iter = 0
    for a in range(len(ALPHA_ARRAY)):
        for x in range(len(HIDDEN_ARRAY_1)):
            for y in range(len(HIDDEN_ARRAY_2)):
                network = NeuralNetworkAckley(
                    INPUT_SIZE_ACKLEY, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_ACKLEY,
                    activation_function=RELU,
                    filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_relu_ackley",
                    iters_check=1
                )
                print(f"RELU ACKLEY Iteration: {iter}")
                network.train_multiple(
                    x_train_ackley, y_train_ackley, ALPHA_ARRAY[a], ITERATIONS,
                    excel_filename="0_1_ackley_relu_excel.xlsx",
                    start_col=iter * 4 + 1 + offset
                )
                network.save_data()
                iter += 1

def process_sigmoid_ackley():
    offset = 15
    iter = 0
    for a in range(len(ALPHA_ARRAY)):
        for x in range(len(HIDDEN_ARRAY_1)):
            for y in range(len(HIDDEN_ARRAY_2)):
                network = NeuralNetworkAckley(
                    INPUT_SIZE_ACKLEY, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_ACKLEY,
                    activation_function=SIGMOID,
                    filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_sigmoid_ackley",
                    iters_check=1
                )
                print(f"SIGMOID ACKLEY Iteration: {iter}")
                network.train_multiple(
                    x_train_ackley, y_train_ackley, ALPHA_ARRAY[a], ITERATIONS,
                    excel_filename="ackley_sigmoid_excel.xlsx",
                    start_col=iter * 4 + 1 + offset
                )
                network.save_data()
                iter += 1

def manual_relu_mnist(alpha, l1, l2):
    offset = 15
    iter = 0
    network = NeuralNetworkMnist(
                    INPUT_SIZE_MNIST, l1, l2, OUTPUT_SIZE_MNIST,
                    activation_function=RELU,
                    filename=f"{alpha}_{l1}_{l2}_{RELU}_mnist",
                    iters_check=1
                )
    print(f"RELU MNIST Iteration: {iter}")
    network.train_multiple(
        x_train, y_train, alpha, ITERATIONS,
        excel_filename=f"end/{alpha}_{l1}_{l2}_{RELU}_mnist_excel.xlsx",
        start_col=iter * 4 + 1 + offset
    )
    network.save_data()
    iter += 1

def manual_sigmoid_mnist(alpha, l1, l2):
    offset = 15
    iter = 0
    network = NeuralNetworkMnist(
                    INPUT_SIZE_MNIST, l1, l2, OUTPUT_SIZE_MNIST,
                    activation_function=SIGMOID,
                    filename=f"{alpha}_{l1}_{l2}_{SIGMOID}_mnist",
                    iters_check=1
                )
    print(f"SIGMOID MNIST Iteration: {iter}")
    network.train_multiple(
        x_train, y_train, alpha, ITERATIONS,
        excel_filename=f"end/{alpha}_{l1}_{l2}_{SIGMOID}_mnist_excel.xlsx",
        start_col=iter * 4 + 1 + offset
    )
    network.save_data()
    iter += 1

def manual_relu_ackley(alpha, l1, l2):
    offset = 15
    iter = 0
    network = NeuralNetworkAckley(
                    INPUT_SIZE_ACKLEY, l1, l2, OUTPUT_SIZE_ACKLEY,
                    activation_function=RELU,
                    filename=None,
                    iters_check=1
                )
    print(f"RELU ACKLEY Iteration: {iter}")
    network.train_multiple(
        x_train_ackley, y_train_ackley, alpha, ITERATIONS,
        excel_filename="merge/update_{alpha}_{l1}_{l2}ackley_sigmoid_excel.xlsx",
        start_col=iter * 4 + 1 + offset
    )
    network.save_data()
    iter += 1

def ackley_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

if __name__ == "__main__":
    # use args in the future for choosing diffrent things, example - activation functions
    # sigmoid / relu / other

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu', help="Choose activation function: relu or sigmoid")
    args = parser.parse_args()
    activation_function = args.activation

    # process_mnist_sigmoid()
    # manual_relu_ackley(0.1, 32, 256)
    # print("MANUAL")
    # print("AUTO")

    # manual_sigmoid_mnist(0.1, 128, 128)
    # manual_sigmoid_mnist(0.1, 128, 256)
    # process_mnist_sigmoid()
    # manual_relu_ackley(0.1, 256, 256)
    # process_relu_ackley()

    # MNIST
    # relu mnist    -   256, 256, 0.1   #
    # sigm mnist    -   256, 256, 0.1

    # ACKLEY
    # relu ackley   -   64, 256, 0.05   ## ok kinda
    # sigm ackley   -   256, 64, 0.1    #

    network = NeuralNetworkAckley(
                    INPUT_SIZE_ACKLEY, 256, 64, OUTPUT_SIZE_ACKLEY,
                    activation_function=SIGMOID,
                    filename="0.1_256_64_sigmoid_ackley",
                    iters_check=1
                )
    
    network.train(x_test_ackley, y_test_ackley, 0.1, 2500+1, None)
    network.save_data()

    # manual_relu_mnist(0.1, 256, 256)
    # manual_sigmoid_mnist(0.1, 256, 256)


