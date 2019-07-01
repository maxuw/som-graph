import torch
import networkx as nx
import random
import numpy as np
import math



class Graph:

    # def __init__(self):

    def create_matrix(self, amount_nodes):
        # jn
        matrix_graph_weights = torch.ones(amount_nodes, amount_nodes)
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


    def search(self, location_vertical, location_horizontal, direction, map_):
        if direction == "l":
            if location_horizontal > 0:
                # print("on map to the direction l: ", map_[location_vertical][location_horizontal - 1])
                if map_[location_vertical][location_horizontal - 1] == 1:
                    ret = location_vertical * len(map_[0]) + location_horizontal - 1
                    # print("searching left ", ret)
                    return ret
                else:
                    return False
            else:
                return False
        elif direction == "u":
            if location_vertical > 0:
                # print("on map to the direction u: ", map_[location_vertical - 1][location_horizontal])
                if map_[location_vertical - 1][location_horizontal] == 1:
                    ret = (location_vertical - 1) * len(map_[0]) + location_horizontal
                    # print("searching up ", ret)
                    return ret
                else:
                    return False
            else:
                return False

        elif direction == "r":
            if location_horizontal < len(map_[0]) - 1:
                # print("on map to the direction r: ", map_[location_vertical][location_horizontal + 1])
                if map_[location_vertical][location_horizontal + 1] == 1:
                    ret = location_vertical * len(map_[0]) + location_horizontal + 1
                    # print("searching right ", ret)
                    return ret
                else:

                    return False
            else:
                return False

        elif direction == "b":
            if location_vertical < (len(map_) - 1):
                # print("on map to the direction b: ", map_[location_vertical + 1][location_horizontal])
                if map_[location_vertical + 1][location_horizontal] == 1:
                    ret = (location_vertical + 1) * len(map_[0]) + location_horizontal
                    # print("searching bottom ", ret)
                    return ret
                else:
                    return False
            else:
                return False
        else:
            return "brak kierunku"


    def look_around(self, location_vertical, location_horizontal, map_):
        directions = ["l", "u", "r", "b"]
        list_results = []
        for direction in directions:
            result = self.search(location_vertical, location_horizontal, direction, map_)
            list_results.append(result)

        return list_results

    def location_special_map(self, num, map_):
        counter_1 = 0
        counter_0 = 0
        for i in range(len(map_)):
            for j in range(len(map_[i])):
                #             print(i, j)

                if map_[i][j] == 1:

                    # print(counter_1)
                    if counter_1 == num:
                        # print("return: ", counter_0, counter_1)
                        return counter_0 + counter_1
                    counter_1 += 1
                else:
                    counter_0 += 1

    def return_axis(self, location_number, map_):
        row = int(location_number / len(map_[0]))
        column = location_number - (row * len(map_[0]))

        return row, column

    # given location on the map returns the node of the graph
    def get_node(self, location_map, map_):
        #     print("location map:", location_map)
        row = int(location_map / len(map_[0]))
        #     print(row)
        #     counter = 0
        count = -1
        for i in range(row + 1):
            #         print(map_[i])
            #         counter = i * len(map[0])
            if i == row:
                #             print("i", i)
                #             print("row", row)

                count += map_[i][0:(location_map - (row * len(map_[0])) +1)].count(1)

            else:
                #             print("else")
                count += map_[i].count(1)
            # print(count)
        return count

    def look_around_by_node_number(self, node_number, map_):
        location_map = self.location_special_map(node_number, map_)
        # print("location map: ", location_map)
        length, width = self.return_axis(location_map, map_)
        # print("length, width: ", length, width)

        return self.look_around(length, width, map_)

    def count_ones(self, map_):
        counter = 0
        for i in range(len(map_)):
            c = map_[i].count(1)
            counter += c

        return counter

    def connect_with_around(self, node_number, map_, matrix, verbose=False):
        # print(graph1)
        list_connections = self.look_around_by_node_number(node_number, map_)

        new_matrix = matrix
        for item in list_connections:
            # print(type(item))
            if type(item) == int:
                target_node = self.get_node(item, map_)
                if verbose==True: print("connecting ", node_number, " ", target_node)
                new_matrix = self.add_edge(new_matrix, node_number, target_node, 0.2)

                list_second_connection = self.look_around_by_node_number(item, map_)

                for second_conn in list_second_connection:
                    if type(second_conn) == int and second_conn != node_number:
                        print(second_conn)
                        target_node = self.get_node(second_conn, map_)
                        print(target_node)
                        if verbose == True: print("making second connection ", node_number, " ", target_node)
                        new_matrix = self.add_edge(new_matrix, node_number, second_conn, 0.4)
        return new_matrix

    def add_small_distance_to_nearby_nodes(self, shape_map_human, distance_matrix, verbose=False):
        new_matrix = distance_matrix
        # print(len(distance_matrix))
        for i in range(len(distance_matrix)):
            # print("iteration", i)
            new_matrix = self.connect_with_around(i, shape_map_human, new_matrix, verbose)

        return new_matrix