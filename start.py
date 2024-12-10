import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
from core.neural_network import NeuralNetwork
from openpyxl import Workbook

# PARAMS
INPUT_SIZE = 784  # 28x28 pixels
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 64
OUTPUT_SIZE = 10  # digits 0 to 9
ALPHA = 0.01
ITERATIONS = 50

# TESTING PARAMS
TEST_BATCH = 20
DEBUG = True

# Tworzenie skoroszytu Excel
workbook = Workbook()
sheet = workbook.active
sheet.title = "Results"
sheet.append(["Iteration", "Training Accuracy", "Test Accuracy", "Prediction", "Actual Label"])

def test_prediction(x_test, y_test, network: NeuralNetwork):
    index = np.random.randint(0, x_test.shape[1] - 1)

    image = x_test[:, index, None]
    label = y_test[index]

    prediction = network.predict(image)

    print(f"Prediction: {prediction[0]}")
    print(f"Actual Label: {label}")

    sheet.append(["", "", "", prediction[0], label])

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

    # Create and train neural network
    network = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE, activation_function=activation_function, filename="mnist_2k_0.01.npz", debug=DEBUG, iters_check=10)
    network.train(x_train, y_train, ALPHA, ITERATIONS, sheet=sheet)
    network.save_data()
    network.plot()



    if DEBUG:
        predictions = network.predict(x_test)
        accuracy = network.get_accuracy(predictions, y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        sheet.append(["", "", accuracy, "", ""])


        for i in range(TEST_BATCH):
            test_prediction(x_test, y_test, network)


    workbook.save("results_mnist.xlsx")
    print("Wyniki zapisane do pliku results.xlsx")