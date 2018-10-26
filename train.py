import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils
import models
import hparam

# Load MNIST Data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = np.reshape(x_train, newshape=[-1, 28, 28, 1]) / 255.0, \
                  np.reshape(x_test, newshape=[-1, 28, 28, 1]) / 255.0

# Set batch_num (number of images divided by batch size)
batch_num = int(x_train.shape[0]/hparam.batch_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # List to save images losses to show results
    list_sample_img = []
    list_subplot = []
    list_loss_G = []
    list_loss_D = []

    # Start train
    for epoch in range(51):

        # train (mb: mini_batch)
        for mb in range(batch_num):
            # Generate noise for generator
            noise = utils.z_noise()

            # Train and compute losses
            l_G, _ = sess.run((models.loss_G, models.optimizer_G),
                              feed_dict={hparam.X_real: x_train[mb * hparam.batch_size: (mb + 1) * hparam.batch_size], hparam.Z_noise: noise,
                                         hparam.is_train: True})
            l_D, _ = sess.run((models.loss_D, models.optimizer_D),
                              feed_dict={hparam.X_real: x_train[mb * hparam.batch_size: (mb + 1) * hparam.batch_size], hparam.Z_noise: noise,
                                         hparam.is_train: True})
            # Append losses
            list_loss_G.append(l_G)
            list_loss_D.append(l_D)

        # Print process
        if epoch % 2 == 0 or epoch == 0:
            print('Epoch:', epoch, ' / G loss:', l_G, ' / D loss:', l_D)

        # get 10 generated samples and save those per specific epoch
        if epoch % 5 == 0:
            sample = sess.run(models.img_generated, feed_dict={hparam.Z_noise: noise, hparam.is_train: False})[0:10]
            sample_reshaped = np.reshape(sample, newshape=[-1, 28, 28])
            list_sample_img.append(sample_reshaped)

    # Set figure to show results
    fig = plt.figure('Generated images')

    # 10 steps of epochs showing results
    for i in range(10):

        # 10 samples per epoch
        for ii in range(10):
            list_subplot.append(fig.add_subplot(10, 10, i * 10 + ii + 1))

            list_subplot[i * 10 + ii].imshow(list_sample_img[i][ii], cmap='gray')
            plt.axis('off')

    plt.figure('Loss')
    plt.plot(list_loss_G, label='loss G')
    plt.plot(list_loss_D, label='loss D', linestyle='--')
    plt.legend()

    plt.show()