#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:49:15 2018

@author: mmvillangca
"""
import plotly.plotly as py
import plotly as plty
from plotly.graph_objs import *
import plotly.figure_factory as FF

import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay

coor = np.loadtxt('insertPointCloudFilenameHere.txt')

x = coor[:,0]
y = coor[:,1]
z = coor[:,2]

points2D = np.vstack([x,y]).T
tri = Delaunay(points2D)
simplices = tri.simplices
cmap = cm.viridis
#R0,G0,B0, alpha0 = cmap(min(z)/(max(z)-min(z)))
#R1,G1,B1, alpha1 = cmap(max(z)/(max(z)-min(z)))


fig1 = FF.create_trisurf(x=x, y=y, z=z,
                         colormap='Viridis',
                         show_colorbar=True,
                         simplices=simplices,
                         height=800, width=1000,
                         title='',
                         showbackground=False, gridcolor='rgb(128, 128, 128)',
                         plot_edges=False, aspectratio=dict(x=1, y=max(y)/max(x), z=max(z)/max(x)))
plty.offline.plot(fig1, filename="insertDesiredFilename")