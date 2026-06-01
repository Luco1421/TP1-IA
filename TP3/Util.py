
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

TYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

class Visualization :
    """! Utility class for data visualization.

    Provides methods for converting tensors into NumPy arrays
    and generating different types of plots for analysis.
    """

    @staticmethod
    def change_tensor(t):
        """ ! Converts a PyTorch tensor into a NumPy array.

        @param t Input object that may be a tensor.
        @return NumPy array if t is a tensor, otherwise the original object.
        """

        # Check whether the input is a PyTorch tensor
        if isinstance(t,torch.Tensor):
            # Detach tensor from graph, move to CPU, and convert to NumPy
            return t.detach().cpu().numpy()
        # Return original object if it is not a tensor
        return np.array(t)

    def make_error_plot(self, history_error, title : str):
        """! Plots the error history over iterations.

        @param history_error List or tensor containing error values.
        @param title Title prefix for the graph.
        """

        # Create a new figure
        plt.figure()

        # Plot error history
        plt.plot(self.change_tensor(history_error))

        # Set plot title
        plt.title(f"{title} - Gráfica de Error")

        # Label X-axis
        plt.xlabel("Iteración")

        # Label Y-axis
        plt.ylabel("Error")

        # Display plot
        plt.show()

    def make_decision_boundary_plot(self, Wo : torch.Tensor, Ws : torch.Tensor, Bo : torch.Tensor, Bs : torch.Tensor , min_x, max_x, min_y, max_y, X_1, X_2):
        """! Plots the decision boundary for a two-dimensional dataset.
        """

        W1 = self.change_tensor(Wo)
        W2 = self.change_tensor(Ws)
        B1 = self.change_tensor(Bo)
        B2 = self.change_tensor(Bs)
        X_1 = self.change_tensor(X_1)
        X_2 = self.change_tensor(X_2)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def prediction(X):
            H = sigmoid((X @ W1) + B1)
            Y = sigmoid((H @ W2) + B2)
            return Y

        xx, yy = np.meshgrid(
            np.linspace(min_x, max_x, 300),
            np.linspace(min_y, max_y, 300)
        )

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        prob = prediction(grid)[:, 0]
        z = prob.reshape(xx.shape)

        plt.figure(figsize=(7,6))

        plt.contourf(xx, yy, z, levels=50, alpha=0.8)

        plt.scatter(
            X_1[:, 0],
            X_1[:, 1],
            c='red',
            s=100,
            label='Clase 0'
        )

        plt.scatter(
            X_2[:, 0],
            X_2[:, 1],
            c='blue',
            s=100,
            label='Clase 1'
        )

        plt.colorbar(label='Salida de la red')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Regiones de decisión XOR')
        plt.show()

    def make_scatter(self, x, y, c):
        """! Generates a scatter plot.

        @param x Values for the X-axis.
        @param y Values for the Y-axis.
        @param c Values used for point coloring.
        """
        # Create a new figure
        plt.figure()

        # Generate scatter plot
        plt.scatter(self.change_tensor(x), self.change_tensor(y), c=self.change_tensor(c))

        # Set plot title
        plt.title("Diagrama de Dipersión")

        # Label X-axis
        plt.xlabel("Eje X")

        # Label Y-axis
        plt.ylabel("Eje Y")

        # Adjust layout spacing
        plt.tight_layout()

        # Display plot
        plt.show()


class Table:
    """! The Table class
    Defines the functions used to generate, update, show and generate a latex of a pandas dataframe
    """

    def __init__(self, data: pd.DataFrame, names : list[str], table_name : str):
        """! Initializes the table object.

        @param data Pandas DataFrame used as table storage.
        @param names List of column names to initialize.
        @param table_name Name used for exporting the table.
        """

        # DataFrame containing the table data
        self.data = data

        # Name used when exporting the table
        self.table_name = table_name

        # Initialize empty columns with object type
        for name in names:
            data[name] = pd.Series(dtype=object)

        self.row_count = 0

    def obtain_row_count(self):
        self.row_count += 1
        return self.row_count - 1

    def add(self, row : int, column : str, value):
        """! Add a value in a specific column of the table
        @param row Row in the table
        @param column Column in the table
        @param value Value to add in the table
        """

        # Rounds float values for consistency
        if isinstance(value, float):
            value = round(value, 6)

        # Inserts or updates the value in the table
        self.data.loc[row, column] = value

    def show(self):
        """! Show the table
        """

        # Show the dataframe
        print(self.data)

    def latex(self, index : bool = False):
        self.data.to_latex(f"{self.table_name}.tex", index=index)

# @brief Dataset generation utilities for machine learning experiments.
class DatasetGenerator:
    """! Utility class for dataset generation.

    Provides helper methods to create synthetic datasets
    commonly used for binary classification experiments.
    """

    @staticmethod
    def generate_nonlinearly_separable(seed, n=10000):
        """! Generates a non-linearly separable dataset.

        @param seed Random seed for reproducibility.
        @return Tuple containing feature tensors and labels.
        """

        # Generate moon-shaped dataset with noise
        x,y = make_moons(n_samples=n, noise=0.08, random_state=seed)

        # Convert dataset to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=TYPE, device = DEVICE)
        y_tensor = torch.tensor(y, dtype=TYPE, device = DEVICE)

        # Return generated dataset
        return x_tensor, y_tensor

    @staticmethod
    def generate_linearly_separable(seed, n=10000):
        """! Generates a linearly separable dataset.

        @param seed Random seed for reproducibility.
        @return Tuple containing feature tensors and labels.
        """

        # Generate clustered dataset
        x, y = make_blobs(n_samples=n, centers=2, random_state=seed, cluster_std=0.35)

        # Convert dataset to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=TYPE, device = DEVICE)
        y_tensor = torch.tensor(y, dtype=TYPE, device = DEVICE)

        # Return generated dataset
        return x_tensor, y_tensor

    @staticmethod
    def split_data(ratio : float, seed : int, points : torch.Tensor, labels : torch.Tensor):
        # Split dataset into training and testing subsets
        x_train, x_test, y_train, y_test = train_test_split(points, labels, test_size = ratio, random_state=seed, shuffle=True)

        # Return training and testing datasets
        return [x_train, y_train], [x_test, y_test]