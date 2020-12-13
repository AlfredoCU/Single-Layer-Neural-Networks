#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:42:07 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

def linear(z, derivative = False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

###############################################################################

def logistic(z, derivative = False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

###############################################################################

def softmax(z, derivative = False):
    e = np.exp(z - np.max(z, axis = 0))
    a = e / np.sum(e, axis = 0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

###############################################################################

class OLN:
    "One-Layer Network"
    
    def __init__(self, n_inputs, n_outputs, activation_function):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_function
    
    def predict(self, X):
        Z = np.dot(self.w, X) + self.b
        return self.f(Z)
    
    def train(self, X, Y, epochs = 200, lr = 0.1):
        p = X.shape[1]
        
        for _ in range(epochs):
            Z = np.dot(self.w, X) + self.b
            Yest, dY = self.f(Z, derivative = True)
            
            lg = (Y - Yest) * dY
            self.w += (lr / p) * np.dot(lg, X.T)
            self.b += (lr / p) * np.sum(lg, axis = 1).reshape(-1, 1)

###############################################################################

# Example

minx = -5
maxx = 5

classes = 8
p_c = 20

X = np.zeros((2, classes * p_c))
Y = np.zeros((classes, classes * p_c))

for i in range(classes):
    seed = minx + (maxx - minx) * np.random.rand(2, 1)
    X[:, i * p_c: (i + 1) * p_c] = seed + 0.15 * np.random.randn(2, p_c)
    Y[i, i * p_c: (i + 1) * p_c] = np.ones((1, p_c))
    
    
net = OLN(2, classes, softmax)
net.train(X, Y, epochs = 300, lr = 1)
Ypred = net.predict(X)

###############################################################################

# Dibujar

cm = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0],
      [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]

ax1 = plt.subplot(1, 2, 1)
y_c = np.argmax(Y, axis = 0)
for i in range(X.shape[1]):
    ax1.plot(X[0, i], X[1, i], "*", c = cm[y_c[i]])
ax1.axis([-5.5, 5.5, -5.5, 5.5])
ax1.set_title("Problema Original")
ax1.grid()

ax2 = plt.subplot(1, 2, 2)
y_c = np.argmax(Ypred, axis = 0)
for i in range(X.shape[1]):
    ax2.plot(X[0, i], X[1, i], "*", c = cm[y_c[i]])
ax2.axis([-5.5, 5.5, -5.5, 5.5])
ax2.set_title("Predicción de la red")
ax2.grid()

plt.savefig("Softmax.eps", format="eps")
