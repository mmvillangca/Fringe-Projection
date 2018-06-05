#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:40:48 2018

@author: mmvillangca
tri-surface from point cloud

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

coor = np.loadtxt('insertPointCloudFileHere.txt')

x = coor[:,0]
y = coor[:,1]
z = coor[:,2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, shade=True)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.zlabel('y',fontsize=14)

plt.show()