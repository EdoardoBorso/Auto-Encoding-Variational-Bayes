import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def dataset_loader(args):
    dataset, _ = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
    dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    dataset = dataset.shuffle(buffer_size=10000).batch(args["batch_size"]).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def plot_and_save_imgs(epoch, iteration, x, recon_x, generated, args, ts):
    plt.figure(figsize=(10, 8))
    for p in range(5):  # Plot 5 pairs of original and reconstructed images
        plt.subplot(4, 5, p + 1)
        plt.imshow(x[p, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Original')

        plt.subplot(4, 5, p + 6)
        plt.imshow(recon_x[p, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Reconstructed')

        plt.subplot(4, 5, p + 11)
        plt.imshow(generated[p, :, :, 0].numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Generated')

    plt.show(block=False)

    if not os.path.exists(os.path.join(args["fig_root"], str(ts))):
        os.makedirs(os.path.join(args["fig_root"], str(ts)))

    plt.savefig(
        os.path.join(args["fig_root"], str(ts),
                     "E{:d}I{:d}.png".format(epoch, iteration)),
        dpi=300)

    plt.clf()
    plt.close('all')


def plot_and_save_dist(epoch, tracker_epoch, ts, args):
    df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
    df['label'] = df['label'].apply(lambda x: int(x))
    g = sns.lmplot(x='x', y='y', hue='label', data=df.groupby('label').head(100), fit_reg=False, legend=True)
    g.savefig(os.path.join(args["fig_root"], str(ts), "E{:d}-Dist.png".format(epoch)), dpi=300)
