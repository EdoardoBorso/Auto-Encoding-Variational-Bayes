import tensorflow as tf
from collections import defaultdict
import time
import vae_model
import losses
import utils


def main(args):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args["learning_rate"])
    vae = vae_model.VAE(
        encoder_layer_sizes=args["encoder_layer_sizes"],
        latent_size=args["latent_size"],
        decoder_layer_sizes=args["decoder_layer_sizes"],
        num_labels=0)

    logs = defaultdict(list)

    ts = time.time()

    dataset = utils.dataset_loader(args)

    for epoch in range(args["epochs"]):
        iterator = iter(dataset)
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration in range(len(dataset)):
            x, y = next(iterator)
            x = tf.reshape(x, [-1, 28, 28, 1])  # Reshape input for Conv2D layers

            with tf.GradientTape() as tape:
                recon_x, mean, log_var, z = vae(x, training=True)
                loss = losses.loss_fn(recon_x, x, mean, log_var)

            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].numpy()
                tracker_epoch[id]['y'] = z[i, 1].numpy()
                tracker_epoch[id]['label'] = yi.numpy()

            logs['loss'].append(loss.numpy())
            if iteration == len(dataset) - 1:
                print(
                    "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch, args["epochs"], iteration, 234,
                                                                                 loss.numpy()))

                z = tf.random.normal([5, args["latent_size"]])
                generated = vae.inference(z, training=False)
                recon_x = tf.reshape(recon_x, [-1, 28, 28, 1])
                generated = tf.reshape(generated, [-1, 28, 28, 1])

                utils.plot_and_save_imgs(epoch, iteration, x, recon_x, generated, args, ts)

        utils.plot_and_save_dist(epoch, tracker_epoch, ts, args)


if __name__ == '__main__':
    args = {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "encoder_layer_sizes": [784, 256],
        "decoder_layer_sizes": [256, 784],
        "latent_size": 4,
        "print_every": 100,
        "fig_root": 'figs'
    }

    main(args)
