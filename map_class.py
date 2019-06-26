import torch
# from torch.nn.modules.distance import PairwiseDistance
from torch.distributions.normal import Normal


class MapClass:

    def __init__(self, length, width, node_dimension, move_closer_coef):
        # print("dupa")
        self.length = length
        self.width = width
        self.node_dimenstion = node_dimension
        self.move_closer_coef = move_closer_coef


        self.weights = self.initialize_weights(self.length, self.width, self.node_dimenstion)
        self.locations = self.initialize_locations(self.weights)
        self.distance_matrix = self.create_distance_matrix(self.locations, self.length, self.width)
        self.impact_matrix = self.calculate_impact_matrix(self.distance_matrix)

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

        print(row, column)
        return(row, column)

    # returns index - topk[1];
    def find_bmu(self, tensor_row_data):
        calc = (self.weights - tensor_row_data).pow(2)
        # print(calc)
        summed_rows = (torch.sum(calc, dim=1))
        # print(summed_rows)
        topk = torch.topk(summed_rows, 1, dim=0, largest=False)
        # print(topk)
        return topk[1]

    def move_closer(self, bmu_index, tensor_row_data):
        change = self.weights[bmu_index] - tensor_row_data

        self.weights[bmu_index].add_(-(change * self.move_closer_coef))

    def initialize_locations(self, weights):
        locations = []
        for i in range(len(weights)):
            location = self.get_location(i)
            locations.append(location)
            # print(location)
        return locations

    def create_distance_matrix(self, locations, length, width):
        distance_matrix = torch.zeros(length * width, length * width)

        for i in range(len(locations)):
            for j in range(i, len(locations)):
                if i != j:
                    tens1 = torch.tensor(locations[i], dtype=torch.float)
                    tens2 = torch.tensor(locations[j], dtype=torch.float)

                    minus = tens1 - tens2
                    minus_power = minus.pow(2)
                    sum_minus_power = torch.sum(minus_power, dim=0)

                    sqrt = torch.sqrt(sum_minus_power)

                    distance_matrix[i][j] = sqrt
                    distance_matrix[j][i] = sqrt

        return distance_matrix

    def calculate_impact_matrix(self, distance_matrix):
        dist = Normal(torch.tensor([0.0]), torch.tensor([2.5]))

        return (dist.cdf(-distance_matrix)) * 2

    def cycle(self, training_data, display_step=False):
        for batch in training_data:
            t_batch = torch.stack([x for x in batch]).float().t()
            for row in t_batch:
                # print(row)
                i_bmu = self.find_bmu(row).item()
                self.move_closer(i_bmu, row)

        # if display_step == True:
        #     basic_visualization(weights_display(weights_.weights))
        #     print(weights_display(weights_.weights))