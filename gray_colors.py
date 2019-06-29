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
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]

# +
# color_names = \
#     ['black', 'blue', 'darkblue', 'skyblue',
#      'greyblue', 'lilac', 'green', 'red',
#      'cyan', 'violet', 'yellow', 'white',
#      'darkgrey', 'mediumgrey', 'lightgrey']
# -



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

data = gray_colors
# data_lables = color_names
batch_size = 2

length = 10
width = 10
number_iterations = 100
shuffle = True

learning_rate = 0.01
# + {}
# trainloader = ""

# def load_data(data, batch_size=4):
#     dim = len(data[0])
#     number_rows_data = len(data)
    
#     trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
#     return trainloader, dim, number_rows_data
# -



map1 = MapClass(data, length, width, learning_rate, number_iterations, matrix1, batch_size=batch_size, shuffle=shuffle)

# training, dim, number_rows_data = load_data(data, batch_size)
map1.basic_visualization()

plt.rcParams['figure.dpi'] = 150
map1.large_cycle(draw_every_epoch=10, rgb=False)


