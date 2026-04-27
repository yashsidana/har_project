import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def find_dataset_dir(base_dir="data"):
    for root, dirs, files in os.walk(base_dir):
        if 'train' in dirs and 'test' in dirs:
            return root
    return None

def load_data(dataset_dir):
    print("Loading data...")
    X_train = pd.read_csv(os.path.join(dataset_dir, 'train', 'X_train.txt'), sep=r'\s+', header=None).values
    y_train = pd.read_csv(os.path.join(dataset_dir, 'train', 'y_train.txt'), sep=r'\s+', header=None).values.ravel() - 1
    
    X_test = pd.read_csv(os.path.join(dataset_dir, 'test', 'X_test.txt'), sep=r'\s+', header=None).values
    y_test = pd.read_csv(os.path.join(dataset_dir, 'test', 'y_test.txt'), sep=r'\s+', header=None).values.ravel() - 1
    
    activity_labels = pd.read_csv(os.path.join(dataset_dir, 'activity_labels.txt'), sep=r'\s+', header=None, index_col=0)
    activity_mapping = activity_labels.to_dict()[1]
    
    return X_train, y_train, X_test, y_test, activity_mapping

# --- Model 1: Base Deep Learning Model (ANN / Simple MLP) ---
class BaseMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BaseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --- Model 2: 1D Convolutional Neural Network (CNN1D) ---
class CNN1DModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(128 * 280, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Model 3: Vanilla Recurrent Neural Network (Simple RNN) ---
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

# --- Model 4: Long Short-Term Memory Network (LSTM) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)

# --- Model 5: Hybrid CNN-LSTM Architecture ---
class HybridCNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
        test_loss = val_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(correct / total)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {correct/total:.4f}")
            
    return train_losses, test_losses, test_accuracies

def plot_results(test_loss_mlp, test_loss_cnn, test_loss_rnn, test_loss_lstm, test_loss_hybrid):
    plt.figure(figsize=(10, 6))
    plt.plot(test_loss_mlp, label='ANN (MLP) Test Loss', linestyle='--')
    plt.plot(test_loss_cnn, label='1D-CNN Test Loss', linestyle='-.')
    plt.plot(test_loss_rnn, label='Simple RNN Test Loss', linestyle=':')
    plt.plot(test_loss_lstm, label='LSTM Test Loss', linestyle='-')
    plt.plot(test_loss_hybrid, label='Hybrid CNN-LSTM Test Loss', linewidth=2)
    plt.title("Loss Curves Comparison (Test Sets)")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()

def plot_confusion_matrix(model, test_loader, activity_mapping, model_name='Hybrid CNN-LSTM', filename='confusion_matrix.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    labels_list = [activity_mapping[i+1] for i in range(len(activity_mapping))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return all_labels, all_preds

def main():
    dataset_dir = find_dataset_dir()
    if not dataset_dir:
        print("Dataset not found. Run download_data.py first.")
        return
        
    X_train, y_train, X_test, y_test, activity_mapping = load_data(dataset_dir)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n--- Training Model 1: ANN (MLP) ---")
    base_model = BaseMLP(input_dim=561, num_classes=6)
    _, mlp_test_loss, _ = train_model(base_model, train_loader, test_loader, epochs=50)
    
    print("\n--- Training Model 2: 1D-CNN ---")
    cnn_model = CNN1DModel(input_dim=561, num_classes=6)
    _, cnn_test_loss, _ = train_model(cnn_model, train_loader, test_loader, epochs=50)

    print("\n--- Training Model 3: Simple RNN ---")
    rnn_model = SimpleRNNModel(input_dim=561, num_classes=6)
    _, rnn_test_loss, _ = train_model(rnn_model, train_loader, test_loader, epochs=50)

    print("\n--- Training Model 4: LSTM ---")
    lstm_model = LSTMModel(input_dim=561, num_classes=6)
    _, lstm_test_loss, _ = train_model(lstm_model, train_loader, test_loader, epochs=50)

    print("\n--- Training Model 5: Hybrid CNN-LSTM ---")
    hybrid_model = HybridCNNLSTM(input_dim=561, num_classes=6)
    _, hybrid_test_loss, _ = train_model(hybrid_model, train_loader, test_loader, epochs=50)
    
    print("\nGenerating Plots...")
    plot_results(mlp_test_loss, cnn_test_loss, rnn_test_loss, lstm_test_loss, hybrid_test_loss)
    true_labels, pred_labels = plot_confusion_matrix(hybrid_model, test_loader, activity_mapping)
    
    print("\n--- Best Model (Hybrid CNN-LSTM) Evaluation ---")
    target_names = [activity_mapping[i+1] for i in range(len(activity_mapping))]
    print(classification_report(true_labels, pred_labels, target_names=target_names))
    print("Done. Plots saved to loss_curves.png and confusion_matrix.png.")

if __name__ == "__main__":
    main()
