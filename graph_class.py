import torch
import networkx as nx
import random
import numpy as np
import math



class Graph:

    # def __init__(self):

    def create_matrix(self, amount_nodes):
        # jn
        matrix_graph_weights = torch.empty(amount_nodes, amount_nodes)
        for i in range(amount_nodes):
            for j in range(amount_nodes):
                matrix_graph_weights[i][j] = 0 if i == j else 1

        return matrix_graph_weights

    def add_edge(self, matrix, length, width, weight):
        # jn
        if length < matrix.shape[0] and width < matrix.shape[0] and width != length:
            matrix[length][width] = weight
            matrix[width][length] = weight

        return matrix

    def generate_edges(self, amount_nodes, percent_edges):
        amount_edges = int((amount_nodes ** 2) * percent_edges)
        print(amount_edges)
        edges = []
        for i in range(amount_edges):
            edge = [random.randint(0, amount_nodes - 1), random.randint(0, amount_nodes - 1), random.random()]
            edges.append(edge)

        return edges

    def add_edges(self, matrix, edges):
        for edge in edges:
            matrix = self.add_edge(matrix, edge[0], edge[1], edge[2])

        return matrix

    def make_list_edges_distances(self, matrix):
        list_edges_weights = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if j != i and not j>i:
                    list_elem = [i, j, matrix[i][j].item()]
        #             print(list_elem)
                    list_edges_weights.append(tuple(list_elem))
        return list_edges_weights

    @staticmethod
    def node_locations(length, width):
        for i in range(length):
            for j in range(width):
                yield np.array([i, j])

    @staticmethod
    def dist(u, v):
        return math.sqrt((u - v).pow(2).sum())

    def standard_som_distance_matrix(self, length, width):
        locations = torch.FloatTensor(np.array(list(self.node_locations(length, width))))
        dists = torch.empty(length * width, length * width)
        for i in range(length * width):
            for j in range(length * width):
                dists[i][j] = self.dist(locations[i], locations[j])

        return dists


    # @staticmethod
    # def node_locations(m, n):
    #     locations = torch.zeros(m, n)
    #     for i in range(m):
    #         for j in range(n):
    #             locations([i, j])

    def build_networkx_graph(self, list_edges_distances):
        g = nx.Graph()
        g.add_weighted_edges_from(list_edges_distances)

        return g