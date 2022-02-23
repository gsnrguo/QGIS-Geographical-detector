#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 22:35
# @Author  : gjg
# @Site    :
# @File    : test_write.py
# @Software: PyCharm

# %% - QGIS
import os
# from qgis._core import QgsProject, QgsApplication
from qgis.core import (QgsVectorLayer, QgsProject)
from qgis.core import QgsClassificationEqualInterval

v_layer = QgsVectorLayer("..data/PM25.shp", "PM25", "ogr")
for field in v_layer.fields():
    print(field.name(), field.typeName())

a = QgsClassificationEqualInterval
a.classes(v_layer, expression='GDP', nclasses=5)

# # vlayer = QgsVectorLayer(path_to_airports_layer, "Airports layer", "ogr")
# if not v_layer.isValid():
#     print("Layer failed to load!")
# else:
#     QgsProject.instance().addMapLayer(v_layer)


# %%

# get the path to a geopackage  e.g. /usr/share/qgis/resources/data/world_map.gpkg
path_to_gpkg = '../data/world_map.gpkg'
gpkg_countries_layer = path_to_gpkg + "|layername=countries"
vlayer = QgsVectorLayer(gpkg_countries_layer, "Countries layer", "ogr")
if not vlayer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(vlayer)

# %% pandas 检测数据类型
import pandas as pd
from pandas.api.types import is_numeric_dtype

df = pd.DataFrame()

# %%
import numpy as np

a = np.array([1, 2, 11, 11, 2])
b = (a == 1)
a[b][0]
# b[True]

# %% geo interval

data = np.array([[1, 3, 4, 5, 2],
                 [2, 3, 1, 6, 3],
                 [1, 5, 2, 3, 1],
                 [3, 4, 9, 2, 1]])

data = whiten(data)

# code book generation
centroids, mean_value = kmeans(data, 3)

print("Code book :\n", centroids, "\n")
print("Mean of Euclidean distances :",
      mean_value.round(4))

clusters, distances = vq(data, centroids)

print("Cluster index :", clusters, "\n")
print("Distance from the centroids :", distances)

# assign centroids and clusters
centroids, clusters = kmeans2(data, 3,
                              minit='random')

print("Centroids :\n", centroids, "\n")
print("Clusters :", clusters)

# %% - kmeans2

from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

rng = np.random.default_rng()
a = rng.multivariate_normal([0, 6], [[2, 1], [1, 1.5]], size=45)
b = rng.multivariate_normal([2, 0], [[1, -1], [-1, 3]], size=30)
c = rng.multivariate_normal([6, 4], [[5, 0], [0, 1.2]], size=25)
z = np.concatenate((a, b, c))
rng.shuffle(z)

centroid, label = kmeans2(z, 3, minit='points')


# %% geometric interval

def geometric(values, classes):
    _min = min(values)
    _max = max(values) + 0.00001  # temporary bug correction: without +0.00001 the max value is not rendered in map
    X = (_max / _min) ** (1 / float(classes))
    res = [_min * X ** k for k in range(classes + 1)]
    return res


res = geometric(np.array([0.179589017, 0.026539462]), 7)
# %%
test = np.array([[0.026539462, 0.046593756, 0.020054, 0],
                 [0.046593757, 0.059616646, 0.013023, 1.539927],
                 [0.059616647, 0.068073471, 0.008457, 1.539927],
                 [0.068073472, 0.081096361, 0.013023, 0.649382],
                 [0.081096362, 0.101150655, 0.020054, 0.649382],
                 [0.101150656, 0.132032793, 0.030882, 0.649382],
                 [0.132032794, 0.179589017, 0.047556, 0.649382]])

# %%
import plotly.express as px

df = px.data.tips()
fig = px.histogram(df, x="total_bill", subplot)
fig.show()

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = ['1970-01-01', '1970-01-01', '1970-02-01', '1970-04-01', '1970-01-02',
     '1972-01-31', '1970-02-13', '1971-04-19']

fig = make_subplots(rows=3, cols=2, subplot_titles=['1','2','3','4','5'])

trace0 = go.Histogram(x=x, nbinsx=4, name='test')
trace1 = go.Histogram(x=x, nbinsx=8)
trace2 = go.Histogram(x=x, nbinsx=10)
trace3 = go.Histogram(x=x,
                      xbins=dict(
                          start='1969-11-15',
                          end='1972-03-31',
                          size='M18'),  # M18 stands for 18 months
                      autobinx=False
                      )
trace4 = go.Histogram(x=x,
                      xbins=dict(
                          start='1969-11-15',
                          end='1972-03-31',
                          size='M4'),  # 4 months bin size
                      autobinx=False
                      )
trace5 = go.Histogram(x=x,
                      xbins=dict(
                          start='1969-11-15',
                          end='1972-03-31',
                          size='M2'),  # 2 months
                      autobinx=False
                      )

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)

fig.show()

#%%
import numpy as np
import pandas as pd
from itertools import combinations
a = np.arange(4)
#
com_a2 = combinations(a,2)

com_a2_list = [i for i in com_a2]
a_com_value = [a[list(i)] for i in com_a2_list]
fuc = np.array([[i.min(),i.max(),i.sum()] for i in a_com_value])
func_df = pd.DataFrame(data=fuc,index=com_a2_list,columns=['inter_min','inter_max','inter_sum'])
func_df
# func_df[['inter_min','inter_sum','inter_max']] - func_df['inter_sum']