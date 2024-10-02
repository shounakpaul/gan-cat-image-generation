# DCGAN for Cat Image Generation

## Introduction
This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch. The model is trained to generate images from a dataset of real images by utilizing a Generator and a Discriminator network. The project follows the GAN (Generative Adversarial Network) structure where the Generator tries to create realistic images while the Discriminator attempts to differentiate between real and generated (fake) images.

## Dataset
I have used the dataset from this link: [Cat Dataset](https://www.kaggle.com/crawford/cat-dataset). 
Download the dataset and put it in the root directory of the project, ie. put all the folders of different types of cats in the `dataset` folder.

## How to use
Firstly, install the required libraries using the following command:
```
pip install -r requirements.txt
```

Run the `main.py` file to train the model. The model will be trained on the cat dataset and will generate images of cats. The generated images will be saved in the `output` folder.

For tuning the hyperparameters, you can change the values in the `config.py` file.

The generated models will be stored in the `output_models` folder.

