Bone Fracture Classifier
------------------------------
Project Overview

This documentation provides a comprehensive overview of the Bone Fracture Classifier model developed by Julia Nuss. The main goal of this project was to create a convolutional neural network (CNN) capable of classifying bone images as fractured or not fractured.
Download link to used dataset: https://drive.google.com/file/d/1WeuxOenviI1_ElW5ISED4MhvR_YFYdmB/view

Main Objectives

Analyze the bone fracture dataset
Develop and train a CNN model for classification
Evaluate the model's performance.
Deploy the model using Flask
Dataset and Image Preprocessing
1.1 Dataset Description

The dataset consists of bone images classified into two categories:
0: Not fractured
1: Fractured

1.2 Image Preprocessing Steps

Grayscaling: Converted images to grayscale to ensure a single channel
Reshaping and Rescaling: Adjusted image size and scaled pixel values
Augmentation:

Horizontal flipping for data variety
Normalization for consistent pixel value ranges
Model Architecture
Model Type: Convolutional Neural Network (CNN)

Layers:

4 Convolutional Layers (Conv2D + MaxPooling)
1 Dropout Layer (to prevent overfitting)
1 Flatten Layer (to convert 2D to 1D)
1 Dense Layer with ReLU activation
Output Layer with Sigmoid activation for binary classification
2.1 Hyperparameters

Activation Functions: ReLU (hidden layers), Sigmoid (output)
Optimizer: Adam
Loss Function: Binary Cross-Entropy
Evaluation Metric: Accuracy

Model Training
The model was trained using the processed dataset.

Labels:
0: Not fractured
1: Fractured

Training Platform: AWS Sagemaker

Model Deployment
Deployment Platform: Flask (Web Application)

Key Steps:

Set up a virtual environment (venv)
Install Flask and its dependencies
Create a Flask app with URL routing for model prediction
The model can be accessed through a web interface for real-time predictions.
5. Model Evaluation

Evaluation Metric: Accuracy

Achieved Accuracy: Ranged from 93.75% to 96.88%

The model was tested and validated on AWS Sagemaker.
6. Future Improvements

Enhancing model with additional augmentation techniques
Experimenting with deeper CNN architectures
Exploring other activation functions (Leaky ReLU, Mish)
Implementing a robust logging system in the Flask application


Developed by Julia Nuss.

