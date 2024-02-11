# Spam Detection with TensorFlow/Keras
## Project Overview
This project aims to develop an automated system for detecting spam messages in texts. 
Utilizing deep learning techniques with TensorFlow and Keras, we focus on creating a model that accurately classifies messages as either spam or ham (non-spam).

## Dataset
The dataset consists of 5,572 messages, pre-labeled as spam or ham, showcasing a diverse range of text data. 


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

## File Descriptions

- AT&T_Spam_Detector.ipynb: The Jupyter notebook containing the data preprocessing, model building, training, and evaluation.
- spam.csv: The dataset used for training and testing the model.
  

## Model Architecture

The project explores two main deep learning models:
- A simple neural network with Global Average Pooling.
- A more sophisticated model utilizing GRU (Gated Recurrent Unit) layers for capturing temporal dependencies in message sequences.

## Results

The GRU-based model demonstrated superior performance, achieving high accuracy on both the training and validation sets. 
Early stopping and other techniques were employed to mitigate overfitting.

## Future Work
Explore alternative model architectures such as LSTM and Transformers.
Implement techniques to handle class imbalance more effectively.
Experiment with more advanced text preprocessing and feature engineering methods.

## Contributing
Your contributions are welcome! Please feel free to send a mail at preuvot.nadia@gmail.com or open an issue for any improvements or bug fixes.# Deep_learning_AT-T-Spam-Detector_Project
