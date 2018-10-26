import tensorflow as tf

import generators
import discriminators
import hparam

# Generated images by generator
img_generated = generators.generator_fc(hparam.Z_noise)

# Discriminated logits of images by discriminator
discriminated_fake = discriminators.discriminator_fc(img_generated)
discriminated_real = discriminators.discriminator_fc(hparam.X_real, reuse=True)

# Loss of generator (fake -> 1 to deceive discriminator)
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_fake, labels=tf.ones_like(discriminated_fake)))
# Loss of discriminator (fake -> 0 and real -> 1 to discriminate images)
loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_fake, labels=tf.zeros_like(discriminated_fake))
                        + tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_real, labels=tf.ones_like(discriminated_real)))

# Divide trainable variables into generator and discriminator
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if "Gen" in var.name]
d_vars = [var for var in t_vars if "Dis" in var.name]

# Optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_G = tf.train.AdamOptimizer(hparam.learning_rate).minimize(loss_G, var_list=g_vars)
    optimizer_D = tf.train.AdamOptimizer(hparam.learning_rate).minimize(loss_D, var_list=d_vars)