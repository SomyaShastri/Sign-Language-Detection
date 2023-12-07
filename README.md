# Sign-Language-Detection
Using Kaggle's Sign Language MNIST dataset, this project employs Convolutional Neural Networks (CNNs) to predict American Sign Language (ASL) characters from images. The focus is on leveraging deep learning to enhance ASL character recognition for practical applications in accessibility and communication.


**Overview:**
This code implements a Convolutional Neural Network (CNN) for predicting characters of American Sign Language (ASL) using the Sign Language MNIST dataset from Kaggle. The dataset comprises images of hand gestures representing ASL characters. The code preprocesses the data, builds a CNN model, trains it, and evaluates its performance on a test set.

**Data Preparation:**
The training and test datasets are loaded from CSV files using pandas.
The labels are extracted, and one-hot encoding is applied to categorize the characters (26 classes).

**Data Processing:**
The image data is reshaped to 28x28 pixels, the standard size for the Sign Language MNIST dataset.
Additional preprocessing includes reshaping the data and normalizing pixel values.

**Model Architecture:**
The CNN model is constructed using the Sequential API from Keras.
The architecture consists of multiple convolutional layers, max-pooling, batch normalization, dropout layers, and densely connected layers.
The model is compiled with the Adam optimizer and categorical crossentropy loss.

**Training**:
The model is trained on the training dataset using a batch size of 50 and for 3 epochs.

**Testing**:
The trained model predicts ASL characters on the test set.
Accuracy is evaluated using the scikit-learn accuracy_score function.

**Results**:
The accuracy score on the test set is calculated and printed.
The predicted labels and actual labels are compared, and accuracy metrics are presented.

**Usage**:
Ensure the required dependencies are installed (pandas, numpy, matplotlib, tensorflow, scikit-learn).
Run the provided code in a Python environment.
Review the model summary, training progress, and test accuracy.
