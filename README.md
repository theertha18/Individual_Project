
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

- **[Model_Training.py](src%2FModel_Training.py)**: Loads preprocessed data, trains multiple machine learning models (Logistic Regression, KNN, Random Forest), evaluates performance, and saves the models.

- **[More_Model_Training.py](src%2FMore_Model_Training.py)**: Loads data, trains additional models including SVM, MLP, CNN, and LSTM, evaluates their performance, and saves the trained models.

- **[Random_Forest.py](src%2FRandom_Forest.py)**: Uses a Random Forest classifier with hyperparameter tuning (GridSearch, RandomSearch, Bayesian, Hyperband) and class balancing with SMOTE.

- **[SVM_Model.py](src%2FSVM_Model.py)**: Classifies signals using SVM with hyperparameter tuning via GridSearchCV and RandomizedSearchCV

- **test_model.py**: Tests a pre-trained model's performance on a new dataset, including data preprocessing and evaluation.

- **[TXT_to_CSV_converter.py](src%2FTXT_to_CSV_converter.py)**: Converts text files to CSV format, aiding in data preprocessing.

- **app.py**: This Flask web application serves as an interface for uploading CSV files containing signals, predicts peaks in the signals using a pre-trained CNN model, and displays the predicted peak positions 
              and distances from peaks on the web page while also calling an external Python script for further processing.
- **index.html**: This HTML template provides a user interface for uploading files, displaying predicted peak positions, windows, and distances from peaks, and rendering a plot image if available.  

## Usage
Each script can be run individually, depending on the specific needs of the analysis or model training process. For example, to run the CNN model training script, use:
```bash
   python CNN_Model.py
```
For GUI, you can run:
```bash
   python app.py
```
Ensure the necessary data files are in the correct directories as expected by each script.
