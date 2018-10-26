import tensorflow as tf

# Hyper parameters
batch_size = 100
noise_dim = 100
learning_rate = 0.0002

# Placeholders
X_real = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
Z_noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, noise_dim])
is_train = tf.placeholder(dtype=tf.bool)