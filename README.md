# Spam Detection with TensorFlow/Keras

## Project Overview
This project develops an automated SMS spam detection system leveraging deep learning techniques within the TensorFlow/Keras framework. The goal is to classify SMS messages into 'spam' or 'ham' (non-spam) accurately, enhancing user security and experience by filtering unwanted messages. We tackle the challenges of natural language processing (NLP) and class imbalance to create a robust model that can be integrated into messaging applications.

## Dataset
The dataset consists of 5,572 messages, pre-labeled as spam or ham, showcasing a diverse range of text data, reflecting a real-world mix of everyday communication and spam content. 

##  Installation
To run this project, you will need to install Python and the following Python libraries:

TensorFlow
Keras
NumPy
Pandas
Matplotlib

You can install these packages using pip:
```
pip install tensorflow keras numpy pandas matplotlib
```
In order to have a well-structured and readable notebook, after opening the notebook, you must click on the small “open colab” tab.

## Preprocessing and Data Cleaning
The preprocessing pipeline includes several steps to prepare the text data for modeling:

* Text Normalization: Converting all messages to lowercase to ensure consistency.
* Tokenization: Splitting text into individual words or tokens to facilitate numerical representation.
* Removing Stop Words and Punctuation: Eliminating common words and punctuation that do not contribute to spam detection.
* Vectorization: Transforming text into numerical vectors that can be fed into the neural network.
* Padding Sequences: Ensuring all input sequences have the same length for batch processing.

## File Descriptions

- AT&T_Spam_Detector.ipynb: The Jupyter notebook containing the data preprocessing, model building, training, and evaluation.
- spam.csv: The dataset used for training and testing the model.
  

## Model Architecture

The project explores two main deep learning models:
* Global Average Pooling Model: A simple neural network model that starts with an embedding layer, followed by a global average pooling layer, and dense layers for classification. This model serves as a baseline to assess the complexity of the problem.

* GRU Model: An advanced model utilizing Gated Recurrent Unit (GRU) layers to better capture temporal dependencies in text sequences. This model includes:

  * An embedding layer for text representation.
  * GRU layers to process the sequence data.
  * Dense layers with a sigmoid activation function for binary classification.

## Results

The GRU-based model outperformed the baseline, showing high accuracy and better handling of sequence data inherent in SMS messages. The implementation of early stopping prevented overfitting, maintaining the model's ability to generalize to unseen data.

## Future Work
Explore alternative model architectures such as LSTM and Transformers.
Implement techniques to handle class imbalance more effectively.
Experiment with more advanced text preprocessing and feature engineering methods.

## Contributing
Your contributions are welcome! Please feel free to send a mail at preuvot.nadia@gmail.com or open an issue for any improvements or bug fixes.# Deep_learning_AT-T-Spam-Detector_Project
