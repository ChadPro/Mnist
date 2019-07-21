# -*- coding=utf-8 -*-
import tensorflow as tf

STDDEV = 0.001
ACTIVATION = tf.nn.relu

class MnistDetect:
    image_size = 28
    output_dim = 10
    default_scope = "Mnist"
    def __init__(self, image_size):
        self.image_size = image_size
    
    def mnist_net(self, inputs):
        with tf.name_scope(self.default_scope):
            net = fc(inputs, 512, "fc1", flatten=True)
            net = fc(net, 256, "fc2", isDropout=True)
            net = fc(net, 10, "out", logistic=True)
        return net

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