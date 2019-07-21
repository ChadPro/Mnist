# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2

""" 以下代码用来获取一些数字图片，作为测试模型识别效果的例子
"""
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

Num = len(train_images)
random_choices = np.random.choice(range(Num), 20, replace=False)
imgs = train_images[random_choices]

for i, img in enumerate(imgs):
    cv2.imwrite("Demos/" + str(i) +".jpg", img)
