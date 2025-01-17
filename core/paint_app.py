from PIL import Image, ImageOps
from io import BytesIO
import tkinter as tk
import numpy as np

from core.neural_network_mnist import NeuralNetworkMnist

class PaintApp:
    def __init__(self, root, nn_model: NeuralNetworkMnist):
        self.root = root
        self.nn_model = nn_model

        self.canvas_size = 280
        self.pixel_size = 10
        self.grid_size = 28  # 28x28 grid
        self.canvas_color = "black"

        self.root.title("28x28 Pixel Painter")
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg=self.canvas_color)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Add a label to show the prediction
        self.prediction_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
        self.prediction_label.pack(pady=10)

        self.image_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    def paint(self, event):
        x, y = event.x, event.y
        grid_x = x // self.pixel_size
        grid_y = y // self.pixel_size

        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            # Update the visual canvas
            self.canvas.create_rectangle(
                grid_x * self.pixel_size,
                grid_y * self.pixel_size,
                (grid_x + 1) * self.pixel_size,
                (grid_y + 1) * self.pixel_size,
                fill="white",
                outline="white"
            )

            # Update the internal image data
            self.image_data[grid_y, grid_x] = 1.0

            # Predict and update the label
            self.update_prediction()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data.fill(0)  # Clear the internal data
        self.prediction_label.config(text="Prediction: None")

    def update_prediction(self):
        # Prepare the image data for the neural network
        input_data = self.image_data.reshape(28 * 28, 1)
        prediction = self.nn_model.predict(input_data)

        # Update the label with the prediction
        self.prediction_label.config(text=f"Prediction: {prediction}")
