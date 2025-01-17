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

def process_mnist_sigmoid():
    offset = 15
    iter = 0
    for a in range(len(ALPHA_ARRAY)):
        for x in range(len(HIDDEN_ARRAY_1)):
            for y in range(len(HIDDEN_ARRAY_2)):
                network = NeuralNetworkMnist(
                    INPUT_SIZE_MNIST, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_MNIST,
                    activation_function=SIGMOID,
                    filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_sigmoid_mnist",
                    iters_check=1
                )
                print(f"SIGMOID MNIST Iteration: {iter}")
                network.train_multiple(
                    x_train, y_train, ALPHA_ARRAY[a], ITERATIONS,
                    excel_filename="mnist_sigmoid_excel.xlsx",
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
                    excel_filename="ackley_relu_excel.xlsx",
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

def ackley_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

def test_prediction(x_test, y_test, network: NeuralNetworkMnist):
    index = np.random.randint(0, x_test.shape[1] - 1)

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

    # def process_mnist_sigmoid():
    #     offset = 15
    #     iter = 0
    #     for a in range(len(ALPHA_ARRAY)):
    #         for x in range(len(HIDDEN_ARRAY_1)):
    #             for y in range(len(HIDDEN_ARRAY_2)):
    #                 network = NeuralNetworkMnist(
    #                     INPUT_SIZE_MNIST, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_MNIST,
    #                     activation_function=SIGMOID,
    #                     filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_sigmoid_mnist",
    #                     iters_check=1
    #                 )
    #                 print(f"SIGMOID MNIST Iteration: {iter}")
    #                 network.train_multiple(
    #                     x_train, y_train, ALPHA_ARRAY[a], ITERATIONS,
    #                     excel_filename="mnist_sigmoid_excel.xlsx",
    #                     start_col=iter * 4 + 1 + offset
    #                 )
    #                 network.save_data()
    #                 iter += 1

    # def process_relu_ackley():
    #     offset = 15
    #     iter = 0
    #     for a in range(len(ALPHA_ARRAY)):
    #         for x in range(len(HIDDEN_ARRAY_1)):
    #             for y in range(len(HIDDEN_ARRAY_2)):
    #                 network = NeuralNetworkAckley(
    #                     INPUT_SIZE_ACKLEY, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_ACKLEY,
    #                     activation_function=RELU,
    #                     filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_relu_ackley",
    #                     iters_check=1
    #                 )
    #                 print(f"RELU ACKLEY Iteration: {iter}")
    #                 network.train_multiple(
    #                     x_train_ackley, y_train_ackley, ALPHA_ARRAY[a], ITERATIONS,
    #                     excel_filename="ackley_relu_excel.xlsx",
    #                     start_col=iter * 4 + 1 + offset
    #                 )
    #                 network.save_data()
    #                 iter += 1

    # def process_sigmoid_ackley():
    #     offset = 15
    #     iter = 0
    #     for a in range(len(ALPHA_ARRAY)):
    #         for x in range(len(HIDDEN_ARRAY_1)):
    #             for y in range(len(HIDDEN_ARRAY_2)):
    #                 network = NeuralNetworkAckley(
    #                     INPUT_SIZE_ACKLEY, HIDDEN_ARRAY_1[x], HIDDEN_ARRAY_2[y], OUTPUT_SIZE_ACKLEY,
    #                     activation_function=SIGMOID,
    #                     filename=f"{ALPHA_ARRAY[a]}_{HIDDEN_ARRAY_1[x]}_{HIDDEN_ARRAY_2[y]}_sigmoid_ackley",
    #                     iters_check=1
    #                 )
    #                 print(f"SIGMOID ACKLEY Iteration: {iter}")
    #                 network.train_multiple(
    #                     x_train_ackley, y_train_ackley, ALPHA_ARRAY[a], ITERATIONS,
    #                     excel_filename="ackley_sigmoid_excel.xlsx",
    #                     start_col=iter * 4 + 1 + offset
    #                 )
    #                 network.save_data()
    #                 iter += 1


    # s_mnist = multiprocessing.Process(target=process_mnist_sigmoid)
    # r_ackley = multiprocessing.Process(target=process_relu_ackley)
    # s_ackley = multiprocessing.Process(target=process_sigmoid_ackley)

    # s_mnist.start()
    # r_ackley.start()
    # s_ackley.start()

    # s_mnist.join()
    # r_ackley.join()
    # s_ackley.join()

    process_relu_ackley()

    # print("ENDED")

    # ACKLEY DATA
    # x_train = np.random.uniform(-2, 2, (INPUT_SIZE, 20000))
    # y_train = ackley_function(x_train[0, :], x_train[1, :]).reshape(1, -1)
    # x_test = np.random.uniform(-2, 2, (INPUT_SIZE, 5000))
    # y_test = ackley_function(x_test[0, :], x_test[1, :]).reshape(1, -1)


    # root = tk.Tk()
    # app = PaintApp(root, network)
    # root.mainloop()


    if False:
        for i in range(TEST_BATCH):
            test_prediction(x_test, y_test, network)
