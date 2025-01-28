# Classification-of-Crops-and-Weeds-Using-CNN

## Project Description

Here is the product aimed on developing a Convolutional Neural Network (CNN) model to accurately classify crops and weeds from agricultural field images. The goal is to assist in automating the process of distinguishing between crop plants and weeds, which is crucial for precision agriculture. By leveraging image recognition techniques through deep learning, the model can help farmers identify and remove weeds efficiently, leading to improved crop yield and reduced use of herbicides.

The CNN model is trained on a dataset of labeled images containing crops and weeds (The dataset can be accessed through the file named "agriculture". The dataset is labelled as "plants" for crop images and "weeds" for weed images. Further they are split into two categories Trian and Test.) 
It processes the images by extracting spatial features at multiple layers and classifies the input into either the "crop" or "weed" category. 
The model aims to achieve high accuracy while being scalable for real-time application in agricultural settings.


## Features
- ### CNN Model for Classification: 
        The project uses a Convolutional Neural Network (CNN) model with multiple layers (Convolution, MaxPooling, Dense, Flatten) to classify images into crops or weeds.

- ### Image Preprocessing and Augmentation: 
        Image preprocessing techniques, such as resizing, normalization, and data augmentation, are applied to improve the model's robustness and accuracy.

- ### Efficient Training and Testing:
        The model is built, trained, and tested with **TensorFlow** and Keras, including optimizations such as the Adam optimizer and cross-entropy loss function.

- ### Model Fine-Tuning:
      The model includes several dense layers for fine-tuning and improved classification accuracy between crops and weeds.

- ### Model Saving and Loading:
    After training, the model is saved as an HDF5 file (.h5 format), thus the **try_one.h5** file is used for prediction in the Flask app.

- ### Real-time Image Classification:
        The saved CNN model is used to classify new images through the **output_check.py** script. It loads an image, preprocesses it, and uses the model to predict whether the image contains crops or weeds.

- ### Flask Web Application:
        A user-friendly web interface built using Flask allows users to upload images and receive classification results in real time. The app routes input images through the model to display results for crops or weeds.

- ### Interactive UI for Predictions:
        The web interface provides dynamic feedback, directing users to appropriate result pages (crops.html or weeds.html) based on the image classification.

- ### Model Deployment:
        The project provides an example of how to deploy a CNN model using Flask, making it accessible via a web browser for real-time image classification.

- ### Support for Multiple Image Formats:
        The project is designed to handle various image formats, preprocess them, and use the model for accurate classification.

## Dataset
The link for dataset: https://justedujo-my.sharepoint.com/:f:/g/personal/inodeh19_eng_just_edu_jo/ErT5Fx_YsMtKoTpvvyckkRcBRERKhQOGBjizgj-0JdOzdA?e=0rBRBE

The dataset used for this project consists of labeled images of crops and weeds captured from agricultural fields. Each image is categorized into one of two classes: crops or weeds. The images vary in size, lighting, and background conditions, simulating real-world agricultural environments.

The dataset is preprocessed using resizing and normalization techniques to ensure uniformity before being fed into the Convolutional Neural Network (CNN). Data augmentation, such as random rotations, flips, and zooms, is applied during training to improve the model's generalization to unseen data. This dataset serves as a crucial component for training the model to accurately distinguish between crops and weeds in real-time.

## Packages Installed
- **TensorFlow**: version==2.2.* For building and training the Convolutional Neural Network (CNN) model.
- **Keras**: For high-level API in TensorFlow to build the CNN architecture.
- **Flask**: For building the web-based front-end interface to allow user interaction with the model.
- **Werkzeug**: For securely handling file uploads in Flask.
- **NumPy**: version==1.19.5 For numerical operations and handling image arrays.
- **OpenCV**: For image processing and transformation (optional).
- **Matplotlib**: For plotting and visualizing the data or training performance (optional).
- **Pillow**: For handling image operations in Python (used by `load_img`).
- **protobuf**: version ==3.20.3

## Example images 
Sample Images of Crop and Weed
Crop 

![Alt Text](https://github.com/Vishweshwar-6/Classification-of-Crops-and-Weeds-Using-CNN/blob/main/Crop_image.jpeg)

Weed 

![Alt Text](https://github.com/Vishweshwar-6/Classification-of-Crops-and-Weeds-Using-CNN/blob/main/Weed_image.jpeg)

## Flask Interface for classification of plants and weeds

![Alt Text](https://github.com/Vishweshwar-6/Classification-of-Crops-and-Weeds-Using-CNN/blob/main/Screenshot%20(58).png)

