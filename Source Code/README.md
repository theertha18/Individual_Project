
# Individual Project - Research on hyperparameters for the classification of ultrasonic signals


## Overview
This project focuses on classifying ultrasonic signals using various machine learning and deep learning models. The aim is to preprocess raw sensor data, perform feature extraction, and build classification models to analyze the data. The core task involves the use of hyperparameter tuning for different models to optimize performance, particularly in detecting objects or features within ultrasonic signals. The project includes the development of a Flask-based web interface to facilitate the uploading of signal data and the visualization of prediction results.

## Installation

 **Clone the repository:**
```bash
   git clone <repository-url>
```
Ensure Python 3.9 or above is installed on your system. You can download it from the official Python website.
Install the required packages:
```bash
   pip install -r requirements.txt
```

## Scripts Description

- **[ADC_FFT_Plot.py](src%2FADC_FFT_Plot.py)**: Processes ADC data and applies FFT for frequency analysis. It includes functionalities for data visualization.

- **[CNN_Model.py](src%2FCNN_Model.py)**:  Trains a 1D CNN on ultrasonic signals with hyperparameter tuning using Keras Tuner.

- **[Create_Test_CSV.py](src%2FCreate_Test_CSV.py)**: Creates a test CSV from the existing dataset.

- **[Data_Preprocessing.py](src%2FData_Preprocessing.py)**: Prepares data by reading CSV files, applying windowing, extracting features (SINAD, peaks, autocorrelation), and saving them as a .npy file.

- **[KNN.py](src%2FKNN.py)**: Classifies signals using KNN with hyperparameter tuning via RandomizedSearchCV

- **[LSTM.py](src%2FLSTM.py):Trains an LSTM model on preprocessed data with hyperparameter tuning using Keras Tuner, optimizing LSTM units, dropout rates, and optimizer.

- **[MLP.py](src%2FMLP.py):Trains a Multi-Layer Perceptron on preprocessed data with hyperparameter tuning using RandomizedSearchCV to optimize architecture, activation, and learning rate strategies.

- **[Model_Training.py](src%2FModel_Training.py)**: Loads preprocessed data, trains multiple machine learning models (Logistic Regression, KNN, Random Forest), evaluates performance, and saves the models.

- **[Model_Training_without_hyperparameters.py](src%2FModel_Training_without_hyperparameters.py)Model_Training.py](src%2FMore_Model_Training.py)**: Loads data, trains additional models including SVM, MLP, CNN, and LSTM without hyperparameter parameters, evaluates their performance, and saves the trained models.

- **[Random_Forest.py](src%2FRandom_Forest.py)**: Uses a Random Forest classifier with hyperparameter tuning (GridSearch, RandomSearch, Bayesian, Hyperband) and class balancing with SMOTE.

- **[SVM_Model.py](src%2FSVM_Model.py)**: Classifies signals using SVM with hyperparameter tuning via GridSearchCV and RandomizedSearchCV

- **[Test_Model.py](src%2FTest_Model.py)**: Tests a pre-trained model's performance on a new dataset, including data preprocessing and evaluation.

- **[TXT_to_CSV_converter.py](src%2FTXT_to_CSV_converter.py)**: Converts text files to CSV format, aiding in data preprocessing.


## Usage
Each script can be run individually, depending on the specific needs of the analysis or model training process. For example, to run the CNN model training script, use:
```bash
   python Test_model.py
```
Ensure the necessary data files are in the correct directories as expected by each script.
