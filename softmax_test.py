# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Alexandre Franco
Date: April 26th, 2018
"""

import numpy as np

scores = np.array([1.0, 2.0, 3.0])
#scores = np.array([[1, 2, 3, 6],
#                   [2, 4, 5, 6],
#                   [3, 8, 7, 6]])

# Compute softmax 
def softmax(x):
#    s = np.exp(x)/np.sum(np.exp(scores), axis=1)
    if x.ndim == 1:
        s = np.exp(x) / np.sum(np.exp(x), axis=0)
    else:
        s = np.exp(x) / np.sum(np.exp(x), axis=0)[None,:]

    return s

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt 

x = np.arange(-2.0 , 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
#scores = x

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()



