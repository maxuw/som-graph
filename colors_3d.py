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

#buildings data
building_sizes = [[0.1, 0.3], [0.1, 0.2], [1., 1.], [0.125, 0.2], [0.529, 0.12], [1.0, 0.3], [0.33, 0.3], 
                  [0.4, 0.4], [0.67, 0.3], [.33, 0.7], [.5, 0.1]]
#     torch.rand((10))

#gray colors data
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]
#     torch.rand((10))

# +
# Network configuration

data = rgb_colors
batch_size = 4

length = 10
width = 10
number_iterations = 20

move_closer_coef = 0.1
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


def large_cycle_rgb(map_, training_data):
    visualize_rgb(map_)
#     print(map_display(map_.map))
    for i in range(number_iterations):
        map_.cycle(training_data)
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

map1 = MapClass(length, width, dim, move_closer_coef, number_iterations)

plt.rcParams['figure.dpi'] = 150
large_cycle_rgb(map1, training)

map1.weights

map1.cycle(training)

map1.distance_matrix

map1.impact_matrix



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


