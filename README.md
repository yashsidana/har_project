# Human Activity Recognition (HAR) Project

## Project Overview
This project classifies daily human activities (such as walking, sitting, standing, etc.) using smartphone accelerometer and gyroscope data from the UCI Machine Learning Repository.

- **Project ID/Group:** P1L
- **Schedule:** Monday 3:30 to 5:10 (Dr. Sushma Jain)
- **Dataset:** [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Models Implemented

| # | Model | Description |
|---|---|---|
| 1 | **Base MLP** | Fully connected neural network baseline |
| 2 | **1D-CNN** | Convolutional filters for local pattern detection |
| 3 | **Simple RNN** | Vanilla recurrent network (demonstrates vanishing gradient) |
| 4 | **LSTM** | Gated recurrent network (solves vanishing gradient) |
| 5 | **Hybrid CNN-LSTM** | CNN feature extraction + LSTM temporal modeling (best model) |

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
```bash
python download_data.py
```

### 3. Train the Baseline Model (Random Forest)
```bash
python train_model.py
```

### 4. Train All Deep Learning Models
```bash
python train_dl_model.py
```

## Results

| Model | Accuracy |
|---|---|
| Simple RNN | 85.30% |
| Base MLP | 88.54% |
| LSTM | 91.20% |
| 1D-CNN | 94.62% |
| **Hybrid CNN-LSTM** | **95.88%** |

## Output Files
- `loss_curves.png` — Validation loss comparison across all 5 models
- `confusion_matrix.png` — Confusion matrix for the best model (Hybrid CNN-LSTM)

## Team
- Yash Sidana (102303973)
- Gaurang Mangla (102303907)
- Hardik Abrol (102303945)
