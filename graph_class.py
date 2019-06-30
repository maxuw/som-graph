import torch
import random

class Graph:

    # def __init__(self):

    def create_matrix(self, amount_vertecies):
        # jn
        matrix_graph_weights = torch.empty(amount_vertecies, amount_vertecies)
        for i in range(amount_vertecies):
            for j in range(amount_vertecies):
                matrix_graph_weights[i][j] = 0 if i == j else 1

        return matrix_graph_weights

    def add_edge(self, matrix, length, width, weight):
        # jn
        if length < matrix.shape[0] and width < matrix.shape[0] and width != length:
            matrix[length][width] = weight
            matrix[width][length] = weight

        return matrix

    def generate_edges(self, amount_vertecies, percent_edges):
        amount_edges = int((amount_vertecies ** 2) * percent_edges)
        print(amount_edges)
        edges = []
        for i in range(amount_edges):
            edge = [random.randint(0, amount_vertecies - 1), random.randint(0, amount_vertecies - 1), random.random()]
            edges.append(edge)

        return edges

    def add_edges(self, matrix, edges):
        for edge in edges:
            matrix = self.add_edge(matrix, edge[0], edge[1], edge[2])

        return matrix