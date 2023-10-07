# Variational Autoencoder (VAE) Implementation

This repository contains an implementation of a Variational Autoencoder (VAE) using TensorFlow. VAE is a generative model that learns to generate new data points similar to a given dataset. It learns a low-dimensional representation of the input data and can be used for various tasks, including image generation, data compression, and anomaly detection.

## Requirements

- Python 3.10
- TensorFlow 2.14.0 
- TensorFlow Datasets 4.9.3
- Matplotlib 3.8.0
- Seaborn 0.13.0
- Pandas 2.1.1

You can install the required packages using `pip`:

```bash
pip install tensorflow tensorflow_datasets matplotlib seaborn pandas
```

# Usage

## Training the VAE

To train the VAE on the MNIST dataset, run the following command:
```bash
python main.py
```

By default, the VAE will be trained for 100 epochs with a batch size of 256 and a latent space dimension of 4. You can modify these parameters and other settings in the `args` dictionary inside the `main` function of `main.py`.

## Results

During training, the VAE's progress will be displayed in the console, showing the current epoch, batch, and loss. After each epoch, a set of original, reconstructed, and generated images will be displayed and saved in the `figs` folder.

Additionally, a scatter plot will be generated, showing the distribution of latent space points for each class in the MNIST dataset. This plot will be saved as `E<epoch>-Dist.png` in the `figs` folder after each epoch.

# File Descriptions
- `main.py`: The main script to train the VAE and visualize results.
- `vae_model.py`: Contains the VAE architecture implemented as a TensorFlow Keras model.
- `losses.py`: Custom loss functions used in training the VAE.
- `utils.py`: Utility functions for data loading and visualization.
- `figs/`: Directory to save generated figures and scatter plots.

# References
- Kingma, D. P., & Welling, M. (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114v10.pdf)

## Credits
This project was inspired by the work in [VAE-CVAE-MNIST](https://github.com/timbmg/VAE-CVAE-MNIST). I adapted their implementation of VAE to work with TensorFlow 2.x.

