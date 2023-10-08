import tensorflow as tf


class VAE(tf.keras.Model):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def call(self, inputs, c=None, training=None):
        if inputs.shape.ndims > 2:
            inputs = tf.reshape(inputs, [-1, 28 * 28])

        means, log_var = self.encoder(inputs, c, training=training)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c, training=training)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def inference(self, z, c=None, training=None):
        return self.decoder(z, c, training=training)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, layer_sizes, latent_size):
        super(Encoder, self).__init__()
        self.MLP = tf.keras.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add(tf.keras.layers.Dense(out_size, activation='relu'))
        self.linear_means = tf.keras.layers.Dense(latent_size)
        self.linear_log_var = tf.keras.layers.Dense(latent_size)

    def call(self, inputs, c=None, training=None):
        x = self.MLP(inputs, training=training)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(tf.keras.layers.Layer):

    def __init__(self, layer_sizes, latent_size):
        super(Decoder, self).__init__()
        self.MLP = tf.keras.Sequential()
        input_size = latent_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            activation = 'sigmoid' if i + 1 == len(layer_sizes) else 'relu'
            self.MLP.add(tf.keras.layers.Dense(out_size, activation=activation))

    def call(self, inputs, c=None, training=None):
        output = self.MLP(inputs, training=training)
        return output
