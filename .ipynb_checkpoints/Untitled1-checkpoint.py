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



shape = \
    [
    [0,1,0],
    [0,1,0],
    [1,1,1]]

from graph_class import Graph
graph1 = Graph()

amount_nodes = graph1.count_ones(shape)

matrix_shape = graph1.create_matrix(amount_nodes)

new_matrix = graph1.add_small_distance_to_nearby_nodes(shape, matrix_shape)

new_matrix

graph1.return_axis(-1, shape)


