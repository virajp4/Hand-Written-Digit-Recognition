# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

## Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to recognize handwritten digits. Handwritten digit recognition is a common computer vision task, and it serves as a fundamental example of how machine learning can be applied to real-world problems.

In this project, we use the popular MNIST dataset, which contains 28x28 pixel grayscale images of handwritten digits (0 to 9) along with their corresponding labels. The primary goal is to train a deep learning model that can accurately classify these images.

The project consists of two main components:

1. **Training a CNN Model**: We use TensorFlow and Keras to build and train a CNN model to recognize handwritten digits from the MNIST dataset. The model is designed to learn and extract features from the input images, making it capable of accurately classifying new, unseen handwritten digits.

2. **Tkinter GUI Application**: To make the digit recognition model accessible and user-friendly, we've created a graphical user interface (GUI) using Tkinter. Users can draw a digit on a canvas, and the model will predict the digit and display the result along with confidence scores.

![digitrec](https://github.com/virajp4/CodeClauseInternship_Hand-Written-Digit-Recognition/assets/122785879/f6890ad9-4bae-4952-8758-e0760ac11b31)

## Project Structure

The project's directory structure is organized as follows:

- `handwritten_digit_recognition.ipynb`: Jupyter Notebook containing the Python code for building and training the CNN model.
- `digit_recog_model.keras`: The trained CNN model saved in Keras format.
- `digit_recog_gui.py`: Python script for the Tkinter-based GUI application.
- `README.md`: This documentation file.

## How the CNN Model Works

The CNN model used in this project is designed to recognize handwritten digits. It consists of several convolutional layers, max-pooling layers, and fully connected layers. Here's a high-level overview of how the model works:

- **Convolutional Layers**: These layers apply convolution operations to the input images, which helps the model learn and extract features from the images. Each convolutional layer consists of multiple filters that slide over the input image, detecting patterns and shapes.

- **Max-Pooling Layers**: After each convolutional layer, max-pooling layers down-sample the feature maps, reducing the spatial dimensions. This helps the model become more invariant to small variations in input images.

- **Fully Connected Layers**: Following the convolutional and max-pooling layers, there are fully connected layers. These layers flatten the feature maps and connect to a traditional feedforward neural network. The final fully connected layer has 10 units, corresponding to the 10 possible digit classes (0-9).

- **Softmax Activation**: The final layer uses the softmax activation function to convert the network's output into probabilities for each digit class. The class with the highest probability is the predicted digit.

## How to Run the Project

To run and interact with this project, follow these steps:

1. Open the Jupyter Notebook `handwritten_digit_recognition.ipynb`. This notebook contains the code for training the CNN model. You can run it cell by cell to load the dataset, build and train the model (OPTIONAL).

2. After training the model, save it as `digit_recog_model.keras`. This file will be used by the Tkinter GUI application. (OPTIONAL)

3. Run the last code block of the Jupyter Notebook to directly start the Tkinter GUI with the pretrained model by me.

4. The GUI application window will open. You can draw a digit on the canvas, click the "Predict" button, and the model will recognize and display the predicted digit along with a confidence score.

## Dependencies

To run this project, you need the following Python libraries and frameworks:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Tkinter (for GUI)

You can install these dependencies using pip command:

```
pip install tensorflow keras numpy matplotlib tkinter
```

## Conclusion

This project demonstrates the use of a CNN to recognize handwritten digits from the MNIST dataset. The trained model achieves high accuracy, and the Tkinter GUI application provides a user-friendly interface for testing the model with hand-drawn digits. Handwritten digit recognition is just one example of how machine learning can be applied to real-world problems, and it serves as a foundation for more complex image recognition tasks.