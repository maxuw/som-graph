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
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt

import random

from map_class import MapClass

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


def create_matrix(amount_vertecies):
    matrix_graph_weights = torch.zeros(amount_vertecies, amount_vertecies)
    for i in range(amount_vertecies):
        matrix_graph_weights[i][i] = 1
        
    return matrix_graph_weights


def add_edge(matrix, length, width, weight):
    if length < matrix.shape[0] and width < matrix.shape[0]:
        matrix[length][width] = weight
        matrix[width][length] = weight
        
    return matrix


def generate_edges(amount_vertecies, percent_edges):
    amount_edges = int((amount_vertecies**2) * percent_edges)
    print(amount_edges)
    edges = []
    for i in range(amount_edges):
        edge = [random.randint(0, amount_vertecies-1), random.randint(0, amount_vertecies-1), random.random()]
        edges.append(edge)
    
    return edges


def add_edges(matrix, edges):
    for edge in edges:
        matrix = add_edge(matrix, edge[0], edge[1], edge[2])
        
    return matrix



# +
# Graph setup
# -

amount_vertecies = 100
percent_edges = 0.5

# +

matrix1 = create_matrix(amount_vertecies)

# +
# matrix1
# -

edges1 = generate_edges(amount_vertecies, percent_edges)

# +
# edges1
# -

matrix1 = add_edges(matrix1, edges1)

# +
# matrix1

# +
# generate_edges(10,0.5)
# -



# +
# Network configuration

data = rgb_colors
data_lables = color_names
batch_size = 4

length = 10
width = 10
number_iterations = 100

learning_rate = 0.01
# + {}
trainloader = ""

def load_data(data, batch_size=4, shuffle=False):
    dim = len(data[0])
    number_rows_data = len(data)
    
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return trainloader, dim, number_rows_data


# -

def large_cycle(map_, training_data):
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))
    for i in range(number_iterations):
        cycle(map_, training_data)
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))


def large_cycle_rgb(map_, training_data, verbose=False):
    visualize_rgb(map_)
#     print(map_display(map_.map))
    for i in range(number_iterations):
        map_.cycle(training_data, verbose)
        if verbose==True: visualize_rgb(map_)
    visualize_rgb(map_)
#     print(map_display(map_.map))


def visualize_rgb(map_):
    print("trying to visualize map")
    tens_try = map_.weights.view(length, width, 3)
    plt.imshow(tens_try)

    classification = map_.classify_all(map_.convert_data_tensor(data))
    for i in range(len(classification)):
        loc_tuple = map_.get_location(classification[i])
        plt.text(loc_tuple[1], loc_tuple[0], color_names[i], ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5, lw=0))

# plt.text(0, 1, color_names[1], ha='center', va='center',
#          bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()



training, dim, number_rows_data = load_data(data)

map1 = MapClass(data, length, width, dim, learning_rate, number_iterations, matrix1, data_lables)

plt.rcParams['figure.dpi'] = 150
map1.large_cycle(training, draw_every_epoch=10, rgb=True)

# +
# plt.rcParams['figure.dpi'] = 150
# large_cycle_rgb(map1, training, verbose=True)
# -

plt.rcParams['figure.dpi'] = 150
large_cycle_rgb(map1, training)

# +
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
frames=200, interval=20, blit=True)
# -

visualize_rgb(map1.weights)

# +

classification

# +

    
    

# +
tens_try = map1.weights.view(length, width, 3)
plt.imshow(tens_try)

classification = map1.classify_all(map1.convert_data_tensor(data))
for i in range(len(classification)):
    loc_tuple = map1.get_location(classification[i])
    plt.text(loc_tuple[1], loc_tuple[0], color_names[i], ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.5, lw=0))
z
# plt.text(0, 1, color_names[1], ha='center', va='center',
#          bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
# -

visualize_rgb(map1)


