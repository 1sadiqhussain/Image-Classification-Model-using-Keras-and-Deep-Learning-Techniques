# Image-Classification-Model-using-Keras-and-Deep-Learning-Techniques

# 📊 CNN Model for MNIST Dataset 🖼️

This Python code demonstrates the creation, training, and evaluation of a Convolutional Neural Network (CNN) model using the MNIST dataset. The code uses TensorFlow and Keras to build and train the model.

## Usage

### 1. Display Sample Images
   - The code initially displays 10 training images from the MNIST dataset in a (2,5) grid of subplots.
   - Image labels are used as titles on the subplots for better visualization.

### 2. Data Preprocessing
   - Images are converted to floats and pixel values are scaled to a range of (0,1).
   - Image shapes are reshaped to (-1, 28, 28, 1) to match the model input requirements.

### 3. CNN Model Architecture
   - The CNN model is built using the Functional API of Keras.
   - It consists of convolutional layers, max-pooling layers, flattening layer, and dense layers.
   - The model summary is displayed.

### 4. Model Compilation
   - The model is compiled using the Adam optimizer, Sparse Categorical Crossentropy loss, and accuracy metric.

### 5. Model Training
   - The model is trained for 25 epochs on the training images and labels.
   - Test images and labels are used as validation data during training.

### 6. Training Visualization
   - Loss and accuracy for both training and validation data are plotted for each epoch.

### 7. Model Evaluation
   - A softmax activation layer is added to the original model to obtain class probabilities.
   - Predictions are made on the test data, and the highest probability class is identified for each test image.

### 8. Result Visualization
   - A plot displays a (5,5) grid of test images, showing ground truth and predicted class labels for each image.

## Prerequisites
   - Python
   - TensorFlow
   - Keras
   - Matplotlib
   - NumPy

## Execution
   - Run the code in a Jupyter Notebook or a Python environment.
   - Ensure necessary libraries are installed using `pip install -r requirements.txt`.

## Author
   - Sadiq Hussain Shaik

Feel free to contribute, provide feedback, or use this code as a learning resource! 🚀
