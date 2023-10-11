# Plant-Disease

This repository contains code for a deep learning model that classifies plant diseases based on the images of plant leaves. The model is built using TensorFlow and Keras, and it is trained and evaluated on a dataset containing 501 images belonging to 5 classes.

![image](https://github.com/Aditya3012Purwar/Plant-Disease/assets/103439955/c3aceae4-6820-4572-a97a-dd93718570f3)

## Dataset

The dataset consists of images of plant leaves which are categorized into five classes:

- Grass Shoots
- Healthy
- Mites
- Ring Spot
- YLD

The dataset is loaded, shuffled, and partitioned into training, validation, and testing datasets.

## Model Architecture

The model uses a sequential architecture including:
- Resizing and rescaling layers
- Data augmentation layers for better generalization
- Four Convolutional layers followed by MaxPooling layers for feature extraction
- Flatten layer to transform the 2D matrix data to a vector
- Dense layers for classification

The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function, and it is evaluated based on its accuracy.

## Dependencies

- TensorFlow
- Keras
- Matplotlib
- NumPy

## Usage

1. Load the dataset from the specified directory using TensorFlow’s image_dataset_from_directory method.
2. Visualize the dataset and understand the distribution of classes.
3. Partition the dataset into training, validation, and testing datasets.
4. Define the model architecture, compile, and train it using the training dataset.
5. Evaluate the model’s performance with the testing dataset.
6. Save the trained model for future use.

## Results

The model's performance is visualized using Matplotlib to plot graphs of accuracy and loss over epochs for both training and validation datasets.

![image](https://github.com/Aditya3012Purwar/Plant-Disease/assets/103439955/511ae2a4-6932-4e74-8aad-0f689426672c)

![image](https://github.com/Aditya3012Purwar/Plant-Disease/assets/103439955/b16afcd3-4e93-4439-9af3-baeafa734774)

## Model Saving

The trained model is saved as a .h5 file which can be loaded later for predictions.

## Prediction

A function `predict` is defined to predict the class of a given image along with the confidence of prediction.

## Visualizing Predictions

Predictions on the testing dataset are visualized, showing the actual and predicted classes along with the confidence of prediction.

![image](https://github.com/Aditya3012Purwar/Plant-Disease/assets/103439955/5e875b0f-35cd-4676-8128-03b73608947d)
