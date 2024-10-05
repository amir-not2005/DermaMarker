# Skin Lesion Detection App

## Overview

This Flask-based web application allows users to upload a photograph of a skin lesion, which is then identified based on the HAM10000 dataset. The app uses a trained machine learning model to classify the lesion into one of seven categories. Due to the large size of the image dataset, Dropbox's API is integrated to store and fetch images as needed.

## Features

- **Skin Lesion Classification**: The app can detect and classify the following types of skin lesions:
  - **nv**: Melanocytic nevi
  - **mel**: Melanoma
  - **bkl**: Benign keratosis-like lesions
  - **bcc**: Basal cell carcinoma
  - **akiec**: Actinic keratoses
  - **vasc**: Vascular lesions
  - **df**: Dermatofibroma

- **Image Upload**: Users can upload an image of a skin lesion for classification.

- **Dropbox Integration**: Large datasets and images are stored and retrieved using the Dropbox API, ensuring efficient image handling.

### Prerequisites

- Python 3.x
- Flask
- Dropbox API SDK
- Machine learning libraries (TensorFlow/PyTorch)
- Other dependencies listed in `requirements.txt`
- VGG16 pretrained model
