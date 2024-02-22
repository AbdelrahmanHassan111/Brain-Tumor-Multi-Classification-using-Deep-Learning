# Brain Tumor Classification using Deep Learning

## Overview
This project focuses on building, training, and evaluating a Convolutional Neural Network (CNN) for the classification of brain tumor images. The CNN is trained on image data to classify brain tumor images into three categories: glioma, meningioma, and pituitary. The model architecture is implemented using TensorFlow and Keras.

---

### Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Brain Tumor Dataset](#brain-tumor-dataset)
9. [Download Dataset](#download-dataset)
10. [License](#license)
11. [Installation](#installation)

---

### 1. Introduction <a name="introduction"></a>
This project utilizes deep learning techniques to develop a model capable of accurately classifying brain tumor images. The model architecture employs convolutional neural networks (CNNs), a popular choice for image classification tasks due to their ability to automatically learn hierarchical features from raw pixel data.

---

### 2. Setup <a name="setup"></a>
The following libraries and packages are used:
- TensorFlow: Open-source machine learning library for building and training neural networks.
- NumPy: Library for numerical computations in Python.
- scikit-learn: Library for machine learning tasks such as classification, regression, and clustering.
- os
- numpy
- matplotlib.pyplot
- tensorflow
- tensorflow.keras.preprocessing.image
- tensorflow.keras.models
- tensorflow.keras.layers
- tensorflow.keras.optimizers
- sklearn.metrics
  
---


### 9. Brain Tumor Dataset <a name="brain-tumor-dataset"></a>
The brain tumor dataset contains T1-weighted contrast-enhanced images from 233 patients, comprising three types of brain tumors: meningioma, glioma, and pituitary tumor. The dataset is organized into MATLAB data format (.mat) files, which have been converted to JPEG format for compatibility with the CNN model.

#### Dataset Details
- Total Images: 3064 slices
- Meningioma: 708 slices
- Glioma: 1426 slices
- Pituitary Tumor: 930 slices
- File Split: Four .zip files, each with 766 slices
- Cross-validation: 5-fold cross-validation indices provided

#### Data Preprocessing
- The dataset is divided into training, validation, and testing sets.
- Data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied using `ImageDataGenerator` from TensorFlow Keras.

---

### 4. Model Architecture <a name="model-architecture"></a>
The CNN model architecture consists of several layers:

#### Convolutional Layers
- The first layer is a Conv2D layer with 32 filters and a kernel size of (3, 3), followed by a Rectified Linear Unit (ReLU) activation function.
- Subsequent Conv2D layers increase the number of filters (64, 128, 256) and maintain the same kernel size and activation function.
- Each Conv2D layer is followed by a MaxPooling2D layer with a pool size of (2, 2) for downsampling.

#### Dense Layers
- After the convolutional layers, the features are flattened using a Flatten layer to prepare them for input to the dense layers.
- Two Dense layers with 512 and 256 units, respectively, are added with ReLU activation functions.
- Dropout layers with a dropout rate of 0.5 are inserted after each dense layer to reduce overfitting.

#### Output Layer
- The final output layer consists of three units, corresponding to the three tumor classes (glioma, meningioma, pituitary).
- The activation function used is softmax, which outputs the probability distribution over the classes.

#### Model Compilation
- The model is compiled using the Adam optimizer with a learning rate of 0.0001.
- Categorical crossentropy is used as the loss function, suitable for multi-class classification problems.
- Accuracy is chosen as the evaluation metric to monitor the model's performance during training.

---

### 5. Training <a name="training"></a>
The model is trained for 45 epochs using the following parameters:
- Batch Size: 32
- Optimizer: Adam with a learning rate of 0.0001
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

**Training Process:**
- Each epoch consists of iterations over the training and validation data.
- The model's performance is evaluated using the training and validation accuracy and loss.
- Data augmentation techniques are applied during training to increase the diversity of the training set and improve generalization.
- Training and validation accuracy and loss are plotted to monitor the model's performance over epochs.

**Training Output:**
Epoch 1/45
67/67 [==============================] - 48s 722ms/step - loss: 0.9789 - accuracy: 0.5140 - val_loss: 1.0762 - val_accuracy: 0.4598
Epoch 2/45
67/67 [==============================] - 32s 484ms/step - loss: 0.9225 - accuracy: 0.5602 - val_loss: 1.1062 - val_accuracy: 0.4554
Epoch 3/45
67/67 [==============================] - 33s 486ms/step - loss: 0.9024 - accuracy: 0.5644 - val_loss: 1.0846 - val_accuracy: 0.4554
...
Epoch 44/45
67/67 [==============================] - 35s 515ms/step - loss: 0.1325 - accuracy: 0.9473 - val_loss: 1.4778 - val_accuracy: 0.6429
Epoch 45/45
67/67 [==============================] - 33s 490ms/step - loss: 0.1638 - accuracy: 0.9370 - val_loss: 1.6051 - val_accuracy: 0.6295

The training process shows a gradual improvement in accuracy over epochs, reaching a peak validation accuracy of 64.29%.

---

### 6. Evaluation <a name="evaluation"></a>

The trained model is evaluated on the validation data to compute various metrics such as loss, accuracy, precision, recall, F1-score, ROC-AUC score, specificity, false positive rate (FPR), false negative rate (FNR), and sensitivity. Classification report and confusion matrix are generated to assess the model's performance.

**Evaluation Metrics:**
- Loss: Categorical crossentropy
- Accuracy: Overall accuracy of the model
- Precision: Ability of the model to avoid false positives
- Recall: Ability of the model to identify true positives
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC Score: Area under the receiver operating characteristic curve
- Specificity: Ability of the model to avoid false positives in binary classification
- False Positive Rate (FPR): Proportion of negative instances incorrectly classified as positive
- False Negative Rate (FNR): Proportion of positive instances incorrectly classified as negative
- Sensitivity: Ability of the model to identify true positives in binary classification

**Classification Report and Confusion Matrix:**
- Detailed report showing precision, recall, and F1-score for each class
- Confusion matrix illustrating the true positive, false positive, true negative, and false negative predictions.

### 7. Results <a name="results"></a>

The evaluation metrics provide insights into the model's performance, including its accuracy in classifying brain tumor images. Various types of accuracy metrics are computed, including overall accuracy and accuracy for each class (glioma, meningioma, and pituitary tumor).

**Results Visualization:**

Classification reports, confusion matrices, and performance metric plots visualize the results of model evaluation.

![output](https://github.com/AbdelrahmanHassan111/Brain-Tumor-Multi-Classification-using-Deep-Learning/assets/156480367/7b0ca303-7fe3-4ca3-acaa-4ce255723d05)

**Training Plots:**

The training accuracy and loss plots provide insights into the model's performance over epochs.
![output2](https://github.com/AbdelrahmanHassan111/Brain-Tumor-Multi-Classification-using-Deep-Learning/assets/156480367/c8c92311-9495-4c06-ad3d-8b6ab7af30fc)

### 8. Conclusion <a name="conclusion"></a>

This project demonstrates the effectiveness of deep learning in classifying brain tumor images. By leveraging CNNs and data augmentation techniques, the model achieves significant accuracy in distinguishing between different tumor types. Further improvements and optimizations can be explored to enhance the model's performance for real-world applications.

### 10. Download Dataset <a name="download-dataset"></a>

You can download the brain tumor dataset from FigShare.
[Download here](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)



### 11. License <a name="license"></a>

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code in accordance with the terms of the license.

### 12. Installation <a name="installation"></a>

You can install the requirements by running:

```bash
pip install -r requirements.txt

#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
