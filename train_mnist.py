# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
from Nets import mnist_conv_net
from Nets import mnist_fc_net
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LEARNING_RATE = 0.001
BATCH_SIZE = 128
SHUFFLE_SIZE = 10000
NUM_CLASS = 10

def train():
    # 输入 placeholder
    images = tf.placeholder(tf.float32, [None, 28, 28, 1], "images")
    labels = tf.placeholder(tf.float32, [None, 10], "labels")

    # 前向传播
    mnist_obj = mnist_conv_net.MnistDetect(image_size=28)
    y = mnist_obj.mnist_net(images)

    # 损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 数据
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    TRAIN_SIZE = len(train_images)
    TEST_SIZE  = len(test_images)
    train_images = np.asarray(train_images, dtype=np.float32) / 255
    train_images = train_images.reshape((TRAIN_SIZE, 28, 28, 1))
    test_images = np.asarray(test_images, dtype=np.float32) / 255
    test_images = test_images.reshape((TEST_SIZE, 28, 28, 1))
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASS).astype(np.float32)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASS).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.shuffle(SHUFFLE_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    oneBatch = dataset.make_one_shot_iterator().get_next()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            data_batch = sess.run(oneBatch)
            sess.run(train_step, feed_dict={images: data_batch[0], labels: data_batch[1]})
            if i % 100 == 0:
                test_acc = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
                print("After %d training step(s), test acc is %g" % (i, test_acc))

        test_acc = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels}) 
        print("Done. Test acc is %g" % test_acc)
        saver.save(sess, "Models/model.ckpt")

def main(argvs):
    train()

if __name__ == "__main__":
    main(sys.argv)