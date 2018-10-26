import tensorflow as tf

import hparam

# Discriminator with fully-connected layers
def discriminator_fc(input, reuse=False):
    with tf.variable_scope('Dis', reuse=reuse):
        input = tf.reshape(input, shape=[-1, 784])
        
        fc_1 = tf.layers.dense(input, units=256)
        leakyrelu_1 = tf.nn.leaky_relu(fc_1)

        fc_2 = tf.layers.dense(leakyrelu_1, units=128)
        leakyrelu_2 = tf.nn.leaky_relu(fc_2)

        fc_3 = tf.layers.dense(leakyrelu_2, units=1)

        output = fc_3

    return output

# Discriminator with CNN
def discriminator_cnn(input, reuse=False):
    with tf.variable_scope('Dis', reuse=reuse):
        conv_1 = tf.layers.conv2d(input, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME')
        leakyrelu_1 = tf.nn.leaky_relu(conv_1)

        conv_2 = tf.layers.conv2d(leakyrelu_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME')
        leakyrelu_2 = tf.nn.leaky_relu(conv_2)

        conv_3 = tf.layers.conv2d(leakyrelu_2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME')
        leakyrelu_3 = tf.nn.leaky_relu(conv_3)

        conv1x1_4 = tf.layers.conv2d(leakyrelu_3, filters=1, kernel_size=[1, 1], strides=[1, 1], padding='SAME')

        gap = tf.reduce_mean(conv1x1_4, axis=[1, 2])

        output = gap

    return output