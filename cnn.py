# -*- coding: utf-8 -*-
"""
    File name: cnn.py
    Author: Theo Chen <theokleintw@gmail.com>
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

LEARNING_RATE = 0.001
EPOCH = 10
NAME = 'cifar10_cnn_ep{}'.format(EPOCH)

# Define Convolutional Neural Network structure
network = input_data(shape=[None, 32, 32, 3], name="input")
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(
    network,
    optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=LEARNING_RATE
)