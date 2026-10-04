# SignLanguage Prediction and Translation

## Project Overview

An image-based Sign Language Recognition and Multilingual Translation application using Deep Learning.

The application recognizes American Sign Language (ASL) hand signs from an uploaded image and converts the recognized sign into text. The recognized text can then be translated into multiple languages using the NLLB-200 multilingual translation model.

## Problem Statement

Develop a Sign Language Translation application for real-time usage using a deep learning-based image recognition model and Natural Language Processing for multilingual translation.

## Objectives

- Recognize ASL signs from images.
- Classify the input image into one of 29 sign classes.
- Convert recognized signs into text.
- Translate the recognized text into multiple languages.
- Provide an easy-to-use Streamlit web application.

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pillow
- Streamlit
- Hugging Face Transformers
- PyTorch
- NLLB-200
- Google Colab
- GitHub

## Dataset

The project uses the ASL Alphabet dataset containing 29 classes:

- A-Z
- del
- nothing
- space

The images are resized to 224 × 224 pixels before being provided to the model.

## Deep Learning Model

EfficientNetB0 is used as the image classification model.

The model uses:

- Pre-trained EfficientNetB0
- ImageNet weights
- Global Average Pooling
- Dense layer with 128 neurons
- Dropout
- Final Dense layer with 29 neurons
- Softmax activation

The final layer produces probability scores for all 29 classes. The class with the highest probability is selected as the predicted sign.

## Prediction Workflow

```text
Input Image
     ↓
Image Preprocessing
     ↓
EfficientNetB0
     ↓
Feature Extraction
     ↓
Classification
     ↓
Predicted Sign
     ↓
Recognized Text