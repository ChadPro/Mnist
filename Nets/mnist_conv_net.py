# -*- coding=utf-8 -*-
import tensorflow as tf

STDDEV = 0.01
ACTIVATION = tf.nn.relu

""" 识别类 Class MnistDetect
"""
class MnistDetect:
    image_size = 28
    output_num = 10
    default_scope = "Mnist"
    def __init__(self, image_size):
        self.image_size = image_size
    
    def mnist_net(self, inputs):
        with tf.name_scope(self.default_scope):
            net = conv2d(inputs, [3,3,1,32], [1,1,1,1], "conv1")
            net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            net = conv2d(net, [3,3,32,64], [1,1,1,1], "conv2")
            net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            net = fc(net, 128, "fc1", flatten=True)
            net = fc(net, self.output_num, "fc2", logistic=True)
        return net

""" 2d 卷积函数
"""
def conv2d(inputs, ksize, strides, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("weight", ksize, initializer=tf.truncated_normal_initializer(STDDEV))
        b = tf.get_variable("bias", ksize[-1], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding="SAME")
        conv = ACTIVATION(tf.nn.bias_add(conv, b))
    return conv

""" fc 全连接函数
"""
def fc(inputs, ksize, scope, logistic=False, flatten=False, isDropout=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if flatten:
            net_shape = inputs.get_shape().as_list()
            nodes = net_shape[1] * net_shape[2] * net_shape[3]
            inputs = tf.reshape(inputs, [-1, nodes]) 
            ksize = [nodes, ksize]
        else:
            ksize = [inputs.shape[-1], ksize]
        w = tf.get_variable("weight", ksize, initializer=tf.truncated_normal_initializer(STDDEV))
        b = tf.get_variable("bias", [ksize[-1]], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(inputs, w) + b

        if isDropout:
            fc = tf.nn.dropout(fc, 0.5)

        if not logistic:
            fc = ACTIVATION(fc)

    return fc
    
