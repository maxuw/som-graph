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

amount_vertecies = 5
percent_edges = 0.5

# +
# using graph class to generate matrix

graph1 = Graph()

# +
# matrix genaration

matrix1 = graph1.create_matrix(amount_vertecies)
edges1 = graph1.generate_edges(amount_vertecies, percent_edges)
matrix1 = graph1.add_edges(matrix1, edges1)
# -

matrix2 = graph1.standard_som_matrix(amount_vertecies, amount_vertecies)

# +
# Network configuration

data = rgb_colors
data_lables = rgb_lables
batch_size = 2


sigma = None
length = 5
width = 5
number_epochs = 100
shuffle = True

learning_rate = 0.01
# -


map1 = MapClass(data, length, width, learning_rate, number_epochs, matrix1, sigma, data_lables, batch_size, shuffle)

# +
# drawtype="rbg" tries to draw colors on map - needs an input data with 3 vectors

# drawtype="black-white" draws black-white
# drawtype="None" - default draws empty space

# labels=True or False draws labels on the map... labels are necessary...
labels=True

drawtype = "rbg"

# draw_every_epoch=0 Don't draw anything
# draw_every_epoch=10 draw every 10 epochs


plt.rcParams['figure.dpi'] = 150
map1.large_cycle(draw_every_epoch=100, drawtype=drawtype, labels=labels)
# -




