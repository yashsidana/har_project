# Human Activity Recognition (HAR) Project

## Project Overview
This project classifies daily human activities (such as walking, sitting, standing, etc.) using the smartphone accelerometer and gyroscope data from the UCI Machine Learning Repository.

- **Project ID/Group:** P1L
- **Schedule:** Monday 3:30 to 5:10 (Dr. Sushma Jain)
- **Dataset:** [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
Run the data downloading script to automatically fetch and extract the dataset from UCI:
```bash
python download_data.py
```

### 3. Train the Model
Run the training script. This will load the data, train a Random Forest classifier, evaluate it on the test set, and save the trained model to a file:
```bash
python train_model.py
```

## Results
The Random Forest model typically achieves over 92% accuracy on the test set using the pre-extracted 561 features provided in the dataset.
