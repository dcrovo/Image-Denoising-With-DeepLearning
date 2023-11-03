# Image-Denoising-With-DeepLearning
A Deep Learning approach to denoise images based on Autoencoders and Denoising Convolutional Neural Networks

## Overview
This project aims to address the issue of noise and deterioration in historical documents when digitized. Historical documents often suffer from poor quality scans, making them hard to read and understand. The goal of this project is to develop two image processing models that can denoise and enhance the quality of scanned documents from historical sources. The objective is to remove noise, stains, wrinkles, and other artefacts that affect the readability of the documents while preserving their historical value and original content.

## Models
This project consists of two main image-denoising models:

### 1. Denoising Autoencoder (DAE)
The Denoising Autoencoder is an unsupervised learning model. It is implemented as a convolutional neural network (CNN) with four convolutional layers in the encoder and decoder. The encoder reduces the dimensionality of the input image and learns to extract relevant features, while the decoder reconstructs the denoised image. The model is trained to minimize the mean squared error between the denoised output and the clean image.

### 2. Denoising Convolutional Neural Network (DnCNN)
The Denoising Convolutional Neural Network is a supervised learning model. It is based on the architecture proposed in the paper "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." The model comprises multiple convolutional layers with batch normalization and residual connections. It is trained to predict the clean image from the noisy input, minimizing the mean squared error loss.

## Data Preparation
The project uses a dataset of historical documents with both noisy and clean images. The data is split into training and validation sets, with transformations such as resizing, rotating, and normalizing applied to augment the training dataset. The noisy images are used as inputs, and the clean images serve as targets for training.

## Hyperparameter Optimization
Hyperparameter optimization was performed to find the best hyperparameters for both the Denoising Autoencoder (DAE) and the Denoising Convolutional Neural Network (DnCNN). The optimization focused on learning rate and feature size for the DAE and learning rate, number of layers, and feature size for the DnCNN. Optuna was used for the optimization process, resulting in improved model performance.

## Results
### Denoising Autoencoder (DAE)
- Without optimization:
  - Peak Signal to Noise Ratio (PSNR) on the validation set: 16.25 dB
  - Mean Squared Error (MSE): 0.0237
- After optimization:
  - Peak Signal to Noise Ratio (PSNR) on the validation set: 18.57 dB (25 epochs)
    
**Here is an example of an image with noise**
![](https://github.com/dcrovo/Image-Denoising-With-DeepLearning/blob/main/preds3_DAE_optimized/y_noisy_0.png)

**Here is an example of the cleaned image**
![](https://github.com/dcrovo/Image-Denoising-With-DeepLearning/blob/main/preds3_DAE_optimized/y_cleaned_0.png)

### Denoising Convolutional Neural Network (DnCNN)
- Without optimization:
  - Peak Signal to Noise Ratio (PSNR) on the validation set: 30.59 dB
  - Mean Squared Error (MSE): 0.00087
- After optimization:
  - Peak Signal to Noise Ratio (PSNR) on the validation set: 28.29 dB (25 epochs)
  - 
**Here is an example of an image with noise**
![](https://github.com/dcrovo/Image-Denoising-With-DeepLearning/blob/main/preds4_DNCNN_optimized/y_noisy_0.png)

**Here is an example of the cleaned image**
![](https://github.com/dcrovo/Image-Denoising-With-DeepLearning/blob/main/preds4_DNCNN_optimized/y_cleaned_0.png)

## Conclusion
The project successfully denoised historical documents using two different models: Denoising Autoencoder (DAE) and Denoising Convolutional Neural Network (DnCNN). After hyperparameter optimization, the models achieved improved performance in terms of PSNR, making the documents more readable and preserving their historical significance.

For details on the implementation and hyperparameter tuning, refer to the Jupyter Notebook in this repository.

