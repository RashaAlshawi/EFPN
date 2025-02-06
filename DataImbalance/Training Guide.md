# Enhanced Feature Pyramid Network (E-FPN) Training Guide

This guide outlines the steps for training the E-FPN model using class decomposition and data augmentation.

## 1. Training with Data Augmentation
To begin training with data augmentation:

- Apply data augmentation to your dataset using the provided code under the "Data Augmentation" section.
- Save the newly augmented dataset and train the E-FPN model on it.

## 2. Class Decomposition
For class decomposition:

- Separate each individual class from your multiclass dataset, ensuring that each class is paired with the correct mask.
- Create a smaller dataset that contains at least three classes.
- Train E-FPN on each smaller dataset and save the trained model.
- Apply ensemble learning to combine the results of the individual models.

## 3. Class Decomposition with Data Augmentation
For class decomposition combined with data augmentation:

- Separate the classes from the augmented dataset created in Step 1.
- Follow the same steps outlined in Step 2 on this augmented dataset.
