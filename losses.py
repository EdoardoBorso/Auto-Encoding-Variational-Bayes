import tensorflow as tf


def loss_fn(recon_x, x, z_mean, z_log_var):
    x_reshaped = tf.reshape(x, [-1, 28 * 28])
    bce = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x_reshaped, recon_x))
    kld = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    kld = tf.reduce_mean(kld)
    return (bce + kld) / 256
