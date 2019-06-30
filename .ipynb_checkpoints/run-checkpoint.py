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

import numpy as np
import matplotlib.pyplot as plt

import random

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

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']





#buildings data
building_sizes = [[0.1, 0.3], [0.1, 0.2], [1., 1.], [0.125, 0.2], [0.529, 0.12], [1.0, 0.3], [0.33, 0.3], 
                  [0.4, 0.4], [0.67, 0.3], [.33, 0.7], [.5, 0.1]]

building_labels = building_sizes



# +
# Graph setup
# -

amount_vertecies = 100
percent_edges = 0.5



graph1 = Graph()
matrix1 = graph1.create_matrix(amount_vertecies)
edges1 = graph1.generate_edges(amount_vertecies, percent_edges)
matrix1 = graph1.add_edges(matrix1, edges1)

# +
# Network configuration

data = building_sizes
data_lables = building_labels
batch_size = 2

length = 10
width = 10
number_epochs = 100
shuffle = True

learning_rate = 0.01
# -


map1 = MapClass(data, length, width, learning_rate, number_epochs, matrix1, data_lables, batch_size, shuffle)

# +
# map1.weights

# +
# training, dim, number_rows_data = load_data(data, batch_size)

# +
# rgb=True tries to draw colors on map - needs an input data with 3 vectors
# rgb=False draws labels on the map... labels are necessary...

plt.rcParams['figure.dpi'] = 150
map1.large_cycle(draw_every_epoch=100, rgb=False)
# -


