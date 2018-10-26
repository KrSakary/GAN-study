import tensorflow as tf

import hparam

# Generator with fully-connected layers
def generator_fc(z):
    with tf.variable_scope('Gen'):
        fc_1 = tf.layers.dense(z, units=256)
        bn_1 = tf.layers.batch_normalization(fc_1, training=hparam.is_train)
        relu_1 = tf.nn.relu(bn_1)

        fc_2 = tf.layers.dense(relu_1, units=512)
        bn_2 = tf.layers.batch_normalization(fc_2, training=hparam.is_train)
        relu_2 = tf.nn.relu(bn_2)

        fc_3 = tf.layers.dense(relu_2, units=784)
        sig_3 = tf.nn.sigmoid(fc_3)
        output = tf.reshape(sig_3, shape=[-1, 28, 28, 1])

    return output

# Generator with DCGAN's guide (except last activation function, not tanh but sigmoid for MNIST)
def generator_DCGAN(z):
    with tf.variable_scope('Gen'):
        fc_1 = tf.layers.dense(z, units= 4 * 4 * 512)
        #bn_1 = tf.layers.batch_normalization(fc_1, training=hparam.is_train)
        #relu_1 = tf.nn.relu(bn_1)
        reshaped_1 = tf.reshape(fc_1, shape=[-1, 4, 4, 512])

        convT_2 = tf.layers.conv2d_transpose(reshaped_1, filters=256, kernel_size=[5, 5], strides=[2, 2], padding='VALID')
        relu_2 = tf.nn.relu(convT_2)
        bn_2 = tf.layers.batch_normalization(relu_2, training=hparam.is_train)
        
        convT_3 = tf.layers.conv2d_transpose(bn_2, filters=128, kernel_size=[5, 5], strides=[2, 2], padding='VALID')
        relu_3 = tf.nn.relu(convT_3)
        bn_3 = tf.layers.batch_normalization(relu_3, training=hparam.is_train)

        convT_4 = tf.layers.conv2d_transpose(bn_3, filters=1, kernel_size=[4, 4], strides=[1, 1], padding='VALID')
        tanh_4 = tf.nn.sigmoid(convT_4)

        output = tanh_4

    return output


