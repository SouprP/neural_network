import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import matplotlib.pyplot as plt
 
import argparse

from core.neural_network import NeuralNetwork


# PARAMS
INPUT_SIZE = 784 # 28x28 pixels
HIDDEN_SIZE = 64 
OUTPUT_SIZE = 10 # digits 0 to 9
ALPHA = 0.01
ITERATIONS = 2500

# TESTING PARAMS
TEST_BATCH = 20
DEBUG = True

# from keras.api.datasets import mnist

def test_prediction(x_test, y_test, network : NeuralNetwork):
    index = np.random.randint(0, x_test.shape[1] - 1)

    image = x_test[:, index, None]
    label = y_test[index]

    prediction = network.predict(image)

    print(f"Prediction: {prediction[0]}")
    print(f"Actual Label: {label}")

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

    network = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, filename="mnist_2.5k_0.01" ,debug=DEBUG, iters_check=10)
    network.train(x_train, y_train, ALPHA, ITERATIONS)
    network.save_data()
    network.plot()

    if DEBUG:
        predictions = network.predict(x_test)
        accuracy = network.get_accuracy(predictions, y_test)
        print(f"Training end accuracy: {accuracy:.4f}")

        for i in range(TEST_BATCH):
            test_prediction(x_test, y_test, network)

