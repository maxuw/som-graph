import torch
# from torch.nn.modules.distance import PairwiseDistance
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt


class MapClass:

    def __init__(self, length, width, node_dimension, learning_rate, number_iterations, matrix_graph_weights):
        # print("dupa")
        self.length = length
        self.width = width
        self.node_dimenstion = node_dimension
        self.learning_rate = learning_rate
        self.number_iterations = number_iterations
        self.matrix_graph_weights = matrix_graph_weights

        self.weights = self.initialize_weights(self.length, self.width, self.node_dimenstion)
        self.locations = self.initialize_locations(self.weights)

        # self.initialize_location(self.length, self.width, self.node_dimenstion)

    def initialize_weights(self, length, width, dimention):
        weights_init = torch.rand((length * width, dimention))


        return weights_init

    def get_location(self, node_number):
        row = "dupa"
        column = "dupa2"

        # if x%width == 0:
        row = int((node_number / self.width))
        column = node_number - (row * self.width)

        # print(row, column)
        return(row, column)

    # returns index - topk[1];
    def find_bmu(self, tensor_row_data, verbose=False):
        calc = (self.weights - tensor_row_data).pow(2)
        # print(calc)
        summed_rows = (torch.sum(calc, dim=1))
        # print(summed_rows)
        topk = torch.topk(summed_rows, 1, dim=0, largest=False)
        if verbose: print(topk[1])
        return topk[1]

    def move_closer(self, bmu_index, tensor_row_data):

        amount_vertecies = self.matrix_graph_weights.shape[0]

        difference = tensor_row_data - self.weights
        change = difference * self.matrix_graph_weights[bmu_index].view(amount_vertecies, 1)
        self.weights = self.weights + (change * self.learning_rate)


    def initialize_locations(self, weights):
        locations = []
        for i in range(len(weights)):
            location = self.get_location(i)
            locations.append(location)
            # print(location)
        return locations


    def step(self, training_data, verbose=False):
        i = 0
        for batch in training_data:
            if i != 0: break
            t_batch = torch.stack([x for x in batch]).float().t()
            row = t_batch[0]
            if verbose: print("row of data", row)
            i_bmu = self.find_bmu(row, verbose).item()
            self.move_closer(i_bmu, row)
            i += 1

        if verbose == True:
            if self.node_dimenstion == 1:
                self.basic_visualization()
                print(self.weights_to_map())
            else:
                self.map_view_for_coding()



    def basic_visualization(self):
        plt.imshow(self.weights_to_map());
        plt.colorbar()
        plt.show()

    def weights_to_map(self): #old map_display
        #     return torch.transpose(map_, 0, 1).view(dim, length, width)
        if self.node_dimenstion == 1:
            return self.weights.view(self.length, self.width)
        else:
            return self.weights.view(self.node_dimenstion, self.length, self.width)

    def map_view_for_coding(self):
        return torch.transpose(self.weights, 0, 1).view(self.node_dimenstion, self.length, self.width)
    #     return map_.view(dim, length, width)

    def classify_all(self, training_data_raw, verbose=False):
        data_classification = []
        for row in training_data_raw:
            # print(row)
            i_bmu = self.find_bmu(row, verbose).item()
            data_classification.append(i_bmu)

        return data_classification

    def convert_data_tensor(self, data):
        list_data_tensor = []
        for row in data:
            row_tensor = torch.tensor(row)
            list_data_tensor.append(row_tensor)

        return list_data_tensor
