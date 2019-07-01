# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch

# import numpy as np
import matplotlib.pyplot as plt

from map_class import MapClass
from graph_class import Graph

#Training inputs for RGBcolors
rgb_colors = [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]]

rgb_lables = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

shape_map_human = \
    [
    [0,0,0,1,0,0,0],
    [1,0,1,1,1,0,1],
    [1,0,0,1,0,0,1],
    [1,1,1,1,1,1,1],
    [0,0,1,1,1,0,0],
    [0,0,1,0,1,0,0],
    [0,0,1,0,1,0,0],
    [0,0,1,0,1,0,0]]
    

special_shape_map

list_zeros.append(2)

list_zeros

map1.history_classifications[0]

rgb_colors = [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],

map1.find_bmu(torch.tensor([0., 0., 0.]))

map1.find_bmu(torch.tensor([1., 1., 1.]))

count zeros between beginning and 6'th element

list_zeros = []
counter = 0
for i in range(len(special_shape_map)):
    for j in range(len(special_shape_map[i])):
        print(i, j)
        if special_shape_map[i][j] == 1:
            

def count_till_number(num):
    counter_1 = 0
    counter_0 = 0
    for i in range(len(special_shape_map)):
        for j in range(len(special_shape_map[i])):
#             print(i, j)
            
            if special_shape_map[i][j] == 1:
                
                print(counter_1)
                if counter_1 == num:
                    print("return")
                    return counter_0 + counter_1
                counter_1 += 1
            else:
                counter_0 += 1


count_till_number(1)#white 45

map1.location_special_map(0)

[map1.location_special_map(x) for x in map1.history_classifications[0]]

map1.history_classifications[1]

# +
# Network configuration

data = rgb_colors
data_lables = rgb_lables
batch_size = 2
special_shape_map = shape_map_human


sigma = None

# for now amount_verticies has to be a multiplication of length and width
length = 8
width = 7
number_epochs = 1000
shuffle = True

learning_rate = 0.01
# -




#buildings data
building_sizes = [[0.1, 0.3], [0.1, 0.2], [1., 1.], [0.125, 0.2], [0.529, 0.12], [1.0, 0.3], [0.33, 0.3], 
                  [0.4, 0.4], [0.67, 0.3], [.33, 0.7], [.5, 0.1]]

building_labels = building_sizes



#Training inputs for RGBcolors
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [0.9], [0.33], [0.4], [0.67], [.33], [.5]]

gray_colors_lables = [[0.1], ["black"], ["white"], [0.125], [0.529], [0.9], [0.33], [0.4], [0.67], [.33], [.5]]

# +
# Graph setup
# -

# for now amount_verticies has to be a multiplication of length and width
amount_nodes = 25
percent_edges = 0.2

# +
# using graph class to generate matrix

graph1 = Graph()

# +
# matrix genaration

matrix1 = graph1.create_matrix(amount_nodes)
edges1 = graph1.generate_edges(amount_nodes, percent_edges)
matrix1 = graph1.add_edges(matrix1, edges1)
list_edges = graph1.make_list_edges_distances(matrix1)
netxgraph1 = graph1.build_networkx_graph(list_edges)
# -

human_matrix = graph1.create_matrix(56)



# +
# matrix1

# +
# This initializes regular SOM grid matrix, it needs to be passes instead of matrix1 for it to work
# Also one needs to experiment with sigma to achieve good results learning rate
# matrix2 = graph1.standard_som_distance_matrix(length, width)
# matrix2
# -



map1 = MapClass(data, length, width, learning_rate, number_epochs, matrix1,
                sigma, data_lables, batch_size, shuffle, netxgraph1, special_shape_map)

# +
# ### Drawing configuration

# drawtype="rbg" tries to draw colors on map - needs an input data with 3 vectors

# drawtype="black-white" draws black-white
# drawtype="networkx" graph drawing using the networkx library
# drawtype="None" - default draws empty space

# Also there is networkx graph drawing

# labels=True or False draws labels on the map... labels are necessary...


# draw_every_epoch=0 Don't draw anything
# draw_every_epoch=10 draw every 10 epochs

# +
# map1.impact_matrix
# -

labels=True
drawtype = "rbg"

# +
# Going through a large cycle combining of number of iteration whole cycles

map1.large_cycle(draw_every_epoch=100, drawtype=drawtype)
# -

# Drawing all the history
plt.rcParams['figure.dpi'] = 150
map1.draw_all(drawtype, labels=labels)

list_zeros = []
counter = 0
for i in range(len(special_shape_map)):
    for j in range(len(special_shape_map[i])):
        print(i, j)
        if special_shape_map[i][j] == 1:
            if counter != 0:
                list_zeros.append(counter)
                print("adding :", counter)
            counter = 0
        else:
            counter += 1

map1.history_classifications

shape_map_human


def look_around(location_vertical, location_horizontal, direction, map_):
    print(location_vertical, location_horizontal)
    search_ = search(location_vertical, location_horizontal, direction, map_)
    if search_ != None and search_ != False :
        print("search object :", search_)
        length = get_node(location_vertical * len(map_[0]) + location_horizontal, map_)
#                         print("go_through_map", i * len(map_[0]) + j)
#                         print(length)
#                         print("search: ", search_)
        width = get_node(search_, map_)
        print("go_through_map", search_)
#                         print(width)
        graph1.add_edge(matrix_distances, length, width, 0.25)
        if length != width:
            print("adding edge ", length, width)


look_around(0, 2, "r", shape_map_human)


def look_around(map_, matrix_distances, matrix_distances):
    directions = ["l", "u", "r", "b"]
    
    for i in range(len(shape_map_human)):
        for j in range(len(shape_map_human[i])):
            if map_[i][j] == 1:


# +
def look_around(location_vertical, location_horizontal, map_):
    directions = ["l", "u", "r", "b"]
    
    for direction in directions:
        search(location_vertical, location_horizontal, direction, map_)
        
                

                    
                    
# -

look_around(1, 6, shape_map_human)



search(7, 3, "b", shape_map_human)


# given location on the map returns the node of the graph
def get_node(location_map, map_):
#     print("location map:", location_map)
    row = int(location_map / len(map_[0]))
#     print(row)
#     counter = 0
    count = 0
    for i in range(row+1):
#         print(map_[i])
#         counter = i * len(map[0])
        if i == row:
#             print("i", i)
#             print("row", row)
            count += map_[i][0:location_map-(row*len(map_[0]))].count(1)
        else:
#             print("else")
            count += map_[i].count(1)
        print(count)
    return count



get_node(11, shape_map_human)

search(3,1, "r", shape_map_human)

go_through_map(shape_map_human)

human_matrix = graph1.create_matrix(56)

human_matrix[0:5]

# +
# go_through_map(shape_map_human, human_matrix)

# +
# human_matrix[0:5]
# -


