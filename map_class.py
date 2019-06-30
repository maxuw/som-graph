import torch
import torch.utils.data
# from torch.nn.modules.distance import PairwiseDistance
# from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import math
import networkx as nx

class MapClass:

################################################
################# Initialization

    def __init__(self, data, length, width, learning_rate, number_iterations,
                 matrix_graph_distances, sigma=None, data_lables=None, batch_size=4,
                 shuffle=True, networkx_graph=None, special_shape_map=None):
        # print("dupa")

        self.amount_nodes = len(matrix_graph_distances)

        # if self.amount_nodes > length*width:
        #     raise NameError('matrix_graph_distances has to equal length*width')
        # if len(matrix_graph_distances.shape) != 2 or matrix_graph_distances.shape[0] != matrix_graph_distances.shape[1]:
        #     raise NameError('invalid matrix_graph_distances')

        self.length = length
        self.width = width
        # self.node_dimenstion = node_dimension
        self.learning_rate = learning_rate
        self.number_iterations = number_iterations
        self.matrix_graph_distances = matrix_graph_distances
        self.classification = None
        self.sigma = sigma

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        # training, dim, number_rows_data
        self.trainloader, self.node_dimenstion, self.number_rows_data = self.load_data(self.data, batch_size=self.batch_size, shuffle=self.shuffle)


        self.data_lables = data_lables
        self.history = []

        self.weights = self.initialize_weights(self.length, self.width, self.node_dimenstion)
        # self.locations = self.initialize_locations(self.weights)

        self.impact_matrix = self.calculate_impact_matrix(self.matrix_graph_distances)


        self.history.append(self.weights.clone())
        # print(self.weights[0:5])
        # print(self.history[0][0:5])

        self.history_classifications = []
        self.history_classifications.append(self.classify_all(self.convert_data_tensor(self.data)))

        # self.initialize_location(self.length, self.width, self.node_dimenstion)
        # self.list_edges = list_edges_distances
        self.netxgraph = networkx_graph
        self.special_shape_map = special_shape_map

################################################
################# Class Functions - Initialize functions


    def load_data(self, data, batch_size, shuffle):
        dim = len(data[0])
        print(dim)
        number_rows_data = len(data)
        print(number_rows_data)

        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        return trainloader, dim, number_rows_data


    def initialize_weights(self, length, width, dimention):
        weights_init = torch.rand((self.amount_nodes, dimention))

        # self.history.append(weights_init)
        return weights_init


    # def initialize_locations(self, weights):
    #     locations = []
    #     for i in range(len(weights)):
    #         location = self.get_location(i)
    #         locations.append(location)
    #         # print(location)
    #     return locations


    def calculate_impact_matrix(self, matrix_graph_distances):
        if self.sigma == None:
            maxv = torch.max(matrix_graph_distances)
            if maxv > 1:
                self.sigma = maxv / 3

            else:
                self.sigma = maxv / 1.5

        sigma2 = self.sigma * self.sigma
        print("sigma: ", self.sigma)
        print("sigma2: ", sigma2)

        impact_matrix = torch.zeros_like(matrix_graph_distances)
        for i in range(self.amount_nodes):
            for j in range(self.amount_nodes):
                impact_matrix[i][j] = self.gaussex(matrix_graph_distances[i][j], sigma2)

        return impact_matrix
        # dist = Normal(torch.tensor([-0.17]), torch.tensor([0.02]))
        # zz = distance_matrix[0]
        # return (dist.cdf(-distance_matrix))

################################################
################# SOM core algorithms

    @staticmethod
    def gaussex(x, sigma2):
        if x == -1:
            return 0
        return math.exp(-x ** 2 / sigma2)

    def get_location(self, node_number):

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
        # if verbose: print(topk[1])
        return topk[1]

    def move_closer(self, bmu_index, tensor_row_data):

        amount_vertecies = self.matrix_graph_distances.shape[0]

        difference = tensor_row_data - self.weights
        change = difference * self.impact_matrix[bmu_index].view(amount_vertecies, 1)
        row_change = (change * self.learning_rate)
        return row_change

    def classify_all(self, training_data_raw, verbose=False):
        data_classification = []
        for row in training_data_raw:
            # print(row)
            i_bmu = self.find_bmu(row, verbose).item()
            data_classification.append(i_bmu)

        return data_classification

################################################
################# Cycle

    def cycle(self, training_data, verbose=False):
        for batch in training_data:
            t_batch = torch.stack([x for x in batch]).float().t()
            # print("batch", batch)
            # print(t_batch.shape)
            # print("t_batch", t_batch)
            batch_change = 0
            for row in t_batch:
                # print(row.shape)
                # print(row)
                i_bmu = self.find_bmu(row, verbose).item()
                sample_change = self.move_closer(i_bmu, row)
                batch_change += sample_change
                # if verbose == True: print("this sample in batch: ", sample_change[0:3])

            self.weights += batch_change
            # if verbose == True: print("this batch change: ", batch_change[0:3])

        # if verbose == True:
        #     self.basic_visualization()
            # print(weights_display(weights_.weights))



    def large_cycle(self, verbose=False, draw_every_epoch=0, drawtype=None, labels=True):

        for i in range(self.number_iterations):
            self.cycle(self.trainloader, verbose)

            # if i % draw_every_epoch == 0 and draw_every_epoch != 0: self.draw_function(drawtype, labels)



        self.history.append(self.weights.clone())
        self.history_classifications.append(self.classify_all(self.convert_data_tensor(self.data)))


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

################################################
################# Visualizations

    def draw_all(self, drawtype="squares", labels=True):
        if drawtype == "networkx":

            self.visualize_networkx(self.netxgraph)

        else:

            for i in range(len(self.history)):
                print("plot of state: ", i)
                self.draw_function(i, drawtype, labels)

    def draw_function(self, history_number, drawtype, labels):

        if drawtype == "rbg": self.visualize_rgb(history_number, labels)
        elif drawtype == "black-white": self.basic_visualization(history_number, labels)
        else:
            if drawtype =="labels":
                self.visualize_norgb(history_number, labels)

    def basic_visualization(self, history_number, labels):
        plt.style.use('grayscale')
        plt.imshow(self.weights_to_map(self.history[history_number]));
        plt.colorbar()

        if labels == True:
            classification = self.history_classifications[history_number]
            for i in range(len(classification)):
                loc_tuple = self.get_location(classification[i])
                plt.text(loc_tuple[1], loc_tuple[0], self.data_lables[i], ha='center', va='center',
                         bbox=dict(facecolor="none", alpha=0.5, lw=0), fontsize=5)

        plt.show()

    def visualize_rgb(self, history_number, labels=True):
        if self.weights.shape[1] == 3:
            if self.special_shape_map != None:
                special_map, special_lables = self.make_special_map(history_number, self.special_shape_map)
                plt.imshow(special_map)
                self.add_lables(history_number, labels, special_lables)
                plt.show()

            elif (self.length*self.width) == self.amount_nodes:

                tens_try = self.history[history_number].view(self.length, self.width, 3)
                plt.imshow(tens_try)




            elif (self.length*self.width) > self.amount_nodes:
                ones_big = torch.ones(self.length*self.width, self.weights.shape[1])
                ones_big[0:self.amount_nodes] = self.history[history_number]
                tens_try = ones_big.view(self.length, self.width, 3)
                plt.imshow(tens_try)

                if labels == True:
                    classification = self.history_classifications[history_number]
                    for i in range(len(classification)):
                        loc_tuple = self.get_location(classification[i])
                        plt.text(loc_tuple[1], loc_tuple[0], self.data_lables[i], ha='center', va='center',
                                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
                plt.show()

            else:
                print("can't run this visualization if there are less length*width against amount nodes")
        else:
            print("you can't visualize this using rgb because the dimension of weights != 3")


    # plt.text(0, 1, color_names[1], ha='center', va='center',
    #          bbox=dict(facecolor='white', alpha=0.5, lw=0))


    #     print(map_display(map_.map))

    def add_lables(self, history_number, labels, special_lables=None):

        if labels == True:
            if special_lables == None:
                classification = self.history_classifications[history_number]
            else:
                classification = special_lables

            for i in range(len(classification)):
                loc_tuple = self.get_location(classification[i])
                print("length classificaion1", len(classification))
                print(classification[i])
                print(loc_tuple)
                print("length data_lables", len(self.data_lables))
                print(self.data_lables[i])
                plt.text(loc_tuple[1], loc_tuple[0], self.data_lables[i], ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.5, lw=0))

    def make_special_map(self, history_number, special_shape_map):
        special_shape_tensor = torch.tensor(special_shape_map)

        if special_shape_tensor.shape != tuple([self.length, self.width]):
            print("to make a special map the special map shape has to match width and length")

        else:
            ones_big = torch.ones(self.length, self.width, self.weights.shape[1])

            tensor_special_map = torch.tensor(special_shape_map)

            p = 0

            for l in range(tensor_special_map.shape[0]):
                for w in range(tensor_special_map.shape[1]):
                    if tensor_special_map[l][w] == 1:
                        # print(w)
                        ones_big[l][w][:] = self.history[history_number][p]
                        # print("added at ", l, " ", w)
                        # print(p)
                        p += 1
            # ones_big.view()

            special_lables = None
            if self.data_lables:
                special_lables = []
                p = 0
                q = 0
                for l in range(tensor_special_map.shape[0]):
                    for w in range(tensor_special_map.shape[1]):
                        if tensor_special_map[l][w] == 1:
                            # print(w)
                            if self.history_classifications[history_number][p] = q:

                                special_lables.append(l* self.width + w)
                                q += 1


                            # print("added at ", l, " ", w)
                            # print(p)
                            p += 1
                # ones_big.view()



            return ones_big, special_lables


    def visualize_norgb(self, history_number, labels=True):

        tens_try = torch.ones(self.length, self.width, 3)
        plt.imshow(tens_try)

        if labels == True:
            classification = self.history_classifications[history_number]
            for i in range(len(classification)):
                loc_tuple = self.get_location(classification[i])
                plt.text(loc_tuple[1], loc_tuple[0], self.data_lables[i], ha='center', va='center',
                         bbox=dict(facecolor="none", alpha=0.5, lw=0), fontsize=5)
        plt.show()

    def visualize_networkx(self, nx_graph, labels=True):
        pos = nx.spring_layout(nx_graph)


        for j in range(len(self.history)):
            classification = self.history_classifications[j]

            labels = dict()

            for i in range(self.amount_nodes):
                labels[i] = str(i)

            for i in range(len(classification)):
                labels[classification[i]] = self.data_lables[i]

            print("plot of state: ", j)
            nx.draw(nx_graph, pos=pos, node_size=(100000/(self.amount_nodes)), with_labels=False)
            nx.draw_networkx_labels(nx_graph, pos, labels)
            plt.draw()
            plt.show()

        # nx.draw(nx_graph, pos=pos, node_size=(10000/self.amount_nodes), with_labels=False)
        # nx.draw_networkx_labels(nx_graph, pos, labels)


        # nx.draw(nx_graph, with_labels=True)
        # plt.draw()
        # plt.show()


        # tens_try = self.weights.view(self.length, self.width, 3)
        # plt.imshow(tens_try)
        #
        # if labels == True:
        #     self.classification = self.classify_all(self.convert_data_tensor(self.data))
        #     for i in range(len(self.classification)):
        #         loc_tuple = self.get_location(self.classification[i])
        #         plt.text(loc_tuple[1], loc_tuple[0], self.data_lables[i], ha='center', va='center',
        #         bbox=dict(facecolor='white', alpha=0.5, lw=0))

    # plt.text(0, 1, color_names[1], ha='center', va='center',
    #          bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.show()

################################################
################# Random


    def weights_to_map(self, weights): #old map_display
        #     return torch.transpose(map_, 0, 1).view(dim, length, width)
        if self.node_dimenstion == 1:
            return weights.view(self.length, self.width)
        else:
            return weights.view(self.node_dimenstion, self.length, self.width)

    def map_view_for_coding(self):
        return torch.transpose(self.weights, 0, 1).view(self.node_dimenstion, self.length, self.width)
    #     return map_.view(dim, length, width)



    def convert_data_tensor(self, data):
        list_data_tensor = []
        for row in data:
            row_tensor = torch.tensor(row)
            list_data_tensor.append(row_tensor)

        return list_data_tensor





