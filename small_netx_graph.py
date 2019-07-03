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


shape_map_human_2 = \
    [
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0]]

# +
# Network configuration

data = rgb_colors
data_lables = rgb_lables
batch_size = 2
special_shape_map = None


sigma = None

# for now amount_verticies has to be a multiplication of length and width
length = 3
width = 3
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
amount_nodes = 6
percent_edges = 0.01

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
matrix1


list_edges

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

map1.draw_all(drawtype="networkx", labels=labels)




