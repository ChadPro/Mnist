# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2

""" train images (60000,28,28) value=[0~255]
    train labels (60000,) value=[0,10]
    test images (10000,28,28)
    test labels (10000)
"""
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)