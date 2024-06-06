"""
Neural network to classify user
In the case study of IF29 class
author : 
"""

import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Connect to MongoDB and import data into pandas
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_label
data = pd.DataFrame(list(collec.find()))

# Features and labels
features = ["verified", "friend_nb", "listed_nb", "follower_nb", 
            "favorites_nb", "len_description", "tweet_nb", "hash_avg", 
            "at_avg", "tweet_user_count", "tweet_frequency", 
            "friend_frequency", "visibility", "Aggressivity"]
X = data[features]
Y = data["label"]

# Encode the labels to be in the range [0, num_classes-1]
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Normalize data
scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
X_sc = scaler.fit_transform(X)

# Remove visibility & aggressivity from training
X_plot = X_sc[["visibility", "Aggressivity"]].copy()
X_sc_removed = X_sc.drop(["visibility", "Aggressivity"], axis=1)

# Slice and correct label
X_train = X_sc_removed[Y_encoded != 1]
Y_train = Y_encoded[Y_encoded != 1]

# Remap labels 2 to 1
Y_train = np.where(Y_train == 2, 1, Y_train)

# Select a subset for training if needed
Y_train = Y_train[:20000]
X_train = X_train.iloc[:20000]

# Check for the number of instances of class 2 (now class 1)
num_twos = (Y_train == 1).sum()
print(f"Number of instances with label 1: {num_twos}")

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.30, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train.values)
x_test = torch.FloatTensor(X_test.values)
Y_train = torch.LongTensor(Y_train).squeeze()
y_test = torch.LongTensor(Y_test).squeeze()

# Verify the unique values in labels to ensure they are in the correct range
print(f"Unique labels in Y_train: {torch.unique(Y_train)}")
print(f"Unique labels in y_test: {torch.unique(y_test)}")

print(f"Shapes - X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {x_test.shape}, Y_test: {y_test.shape}")

class Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_classes):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_model(X_train, Y_train, x_test, y_test, model, criterion, optimizer, n_epochs, batch_size):
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    losses = []
    accuracies_train = []
    accuracies_test = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        train_accuracy = correct_train / total_train
        accuracies_train.append(train_accuracy)
        
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()
            test_accuracy = correct_test / total_test
            accuracies_test.append(test_accuracy)
        
        losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Loss: {losses[-1]:.4f}")

    return losses, accuracies_train, accuracies_test

# Define model parameters
input_size = X_train.shape[1]
hidden_sizes = [45, 20, 10]
output_classes = 2 
model = Model(input_size, hidden_sizes, output_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
n_epochs = 30
batch_size = 64
losses, accuracies_train, accuracies_test = train_model(X_train, Y_train, x_test, y_test, model, criterion, optimizer, n_epochs, batch_size)

# Plot loss and accuracies
plt.plot(np.arange(1, n_epochs+1), losses, label='Loss')
plt.plot(np.arange(1, n_epochs+1), accuracies_train, label='Train Accuracy')
plt.plot(np.arange(1, n_epochs+1), accuracies_test, label='Test Accuracy')
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.legend()
plt.show()


# Visualize the classification results
def plot_classification_results(X_plot, predictions):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_plot['visibility'], X_plot['Aggressivity'], c=predictions, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ticks=[0, 1])
    plt.xlabel('Visibility')
    plt.ylabel('Aggressivity')
    plt.title('Classification Results')
    plt.show()

# Get predictions for the entire dataset
model.eval()
X_full = torch.FloatTensor(X_sc_removed.values)
with torch.no_grad():
    outputs = model(X_full)
    _, predictions = torch.max(outputs, 1)

# Plot the classification results
plot_classification_results(X_plot, predictions.numpy())
