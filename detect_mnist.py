# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
from Nets import mnist_conv_net
from Nets import mnist_fc_net
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #不关健，只是屏蔽一些输出，看起来好看

images = tf.placeholder(tf.float32, [1, 28, 28, 1], "images")

mnist_obj = mnist_conv_net.MnistDetect(image_size=28)
y = mnist_obj.mnist_net(images)

with tf.Session() as sess:
    moder_saver = tf.train.Saver()
    moder_saver.restore(sess, "Models/model.ckpt")

    Demos = [("Demos/" + str(i) + ".jpg") for i in range(20)]

    detect_result = []
    for demo_path in Demos:
        img = cv2.imread(demo_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, -1)

        result = sess.run(y, feed_dict={images:img})
        result = np.argmax(result)
        detect_result.append(result)
    
    print("##### 识别Demos中的图片结果如下 #####")
    print(detect_result)