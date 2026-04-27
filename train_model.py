import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def find_dataset_dir(base_dir="data"):
    # Look for the directory containing 'train' and 'test' folders
    for root, dirs, files in os.walk(base_dir):
        if 'train' in dirs and 'test' in dirs:
            return root
    return None

def load_data(dataset_dir):
    print("Loading training data...")
    X_train = pd.read_csv(os.path.join(dataset_dir, 'train', 'X_train.txt'), sep=r'\s+', header=None)
    y_train = pd.read_csv(os.path.join(dataset_dir, 'train', 'y_train.txt'), sep=r'\s+', header=None).values.ravel()
    
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(dataset_dir, 'test', 'X_test.txt'), sep=r'\s+', header=None)
    y_test = pd.read_csv(os.path.join(dataset_dir, 'test', 'y_test.txt'), sep=r'\s+', header=None).values.ravel()
    
    # Load activity labels
    activity_labels = pd.read_csv(os.path.join(dataset_dir, 'activity_labels.txt'), sep=r'\s+', header=None, index_col=0)
    activity_mapping = activity_labels.to_dict()[1]
    
    return X_train, y_train, X_test, y_test, activity_mapping

def main():
    dataset_dir = find_dataset_dir()
    if not dataset_dir:
        print("Error: Could not find the dataset directory. Make sure you have run download_data.py first.")
        return

    print(f"Dataset found at: {dataset_dir}")
    X_train, y_train, X_test, y_test, activity_mapping = load_data(dataset_dir)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    target_names = [activity_mapping[i] for i in sorted(activity_mapping.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    model_path = "har_random_forest_model.joblib"
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Done!")

if __name__ == "__main__":
    main()
