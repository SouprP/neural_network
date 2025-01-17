import numpy as np
import matplotlib.pyplot as plt
import argparse
from core.neural_network_ackley import NeuralNetworkAckley
from openpyxl import Workbook

# Parametry sieci
INPUT_SIZE = 2
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 64
OUTPUT_SIZE = 1
ALPHA = 0.01
ITERATIONS = 80
DEBUG = True
ITERS_CHECK = 10

# Tworzenie skoroszytu Excel
workbook = Workbook()
sheet = workbook.active
sheet.title = "Ackley Results"
sheet.append(["Iteration", "MSE", ])

# Funkcja Ackleya
def ackley_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

# Generowanie siatki punktów
def generate_grid(x_min, x_max, step):
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(x_min, x_max + step, step)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    return grid, X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu')
    args = parser.parse_args()


    x_train = np.random.uniform(-2, 2, (INPUT_SIZE, 20000))
    y_train = ackley_function(x_train[0, :], x_train[1, :]).reshape(1, -1)
    x_test = np.random.uniform(-2, 2, (INPUT_SIZE, 5000))
    y_test = ackley_function(x_test[0, :], x_test[1, :]).reshape(1, -1)


    network = NeuralNetworkAckley(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE, activation_function=args.activation, filename="ackley_relu_2k_0.01", debug=DEBUG, iters_check=ITERS_CHECK)
    network.train(x_train, y_train, alpha=ALPHA, iterations=ITERATIONS, sheet=sheet)

    grid, X, Y = generate_grid(-2, 2, 0.01)
    predictions = network.predict(grid.T).reshape(X.shape)
    actual = ackley_function(X, Y)

    mse = network.get_mse(predictions, actual)
    print(f"Train MSE: {mse:.4f}")
    sheet.append(["Train", mse])

    # Wizualizacja wyników
    plt.figure(figsize=(12,8))
    plt.contourf(X, Y, actual, levels=50, cmap='viridis', alpha=1.0)
    plt.colorbar(label='Function actual Value')
    plt.contour(X, Y, predictions, levels=40, cmap='cool', alpha=0.3)
    plt.colorbar(label='Value')
    plt.title("Ackley Function Approximation - train")
    plt.show()

    # Wykres 3D rzeczywistej funkcji Ackleya
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, actual, cmap='viridis', alpha=0.9)
    ax.set_title("Ackley Function - True Values (Train Set)")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Ackley Value')

    # Wykres 3D przewidywań sieci
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, predictions, cmap='viridis', alpha=0.9)
    ax2.set_title("Ackley Function Approximation - train")
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Predicted Value')
    plt.show()

    # Wykres MSE w czasie iteracji
    network.plot_mse()

    predictions_test = network.predict(x_test)
    mse_test = network.get_mse(predictions_test, y_test)
    print(f"Test MSE: {mse_test:.4f}")
    sheet.append(["Test", mse_test])
    workbook.save("ackley_results1.xlsx")
    print("Wyniki zapisane do pliku ackley_results.xlsx")

    # Generowanie siatki dla wizualizacji
    grid, X_test, Y_test = generate_grid(-2, 2, 0.01)
    predictions_test_grid = network.predict(grid.T).reshape(X_test.shape)

    # Wizualizacja wyników testowych (2D)
    plt.figure(figsize=(12, 8))
    plt.contourf(X_test, Y_test, ackley_function(X_test, Y_test), levels=50, cmap='viridis', alpha=0.9)
    plt.colorbar(label='Value')
    plt.contour(X_test, Y_test, predictions_test_grid, levels=30, cmap='cool', alpha=0.6)
    plt.colorbar(label='Value')
    plt.title("Ackley Function Approximation on Test Set")
    plt.show()

    # Wizualizacja wyników testowych (3D)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X_test, Y_test, ackley_function(X_test, Y_test), cmap='viridis', alpha=0.9)
    ax.set_title("Ackley Function - True Values (Test Set)")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Ackley Value')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_test, Y_test, predictions_test_grid, cmap='viridis', alpha=0.9)
    ax2.set_title("Ackley Function Approximation - Test")
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Predicted Value')
    plt.show()