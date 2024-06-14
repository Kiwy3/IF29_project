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
features = ['verified', 'protected', 'friend_nb',
        'listed_nb', 'follower_nb', 'favorites_nb', 'len_description',
        'hash_avg', 'mention_avg', 'url_avg', 'symbols_avg', 'tweet_nb',
        'tweet_user_count', 'user_lifetime', 'tweet_frequency',
        'friend_frequency', 'aggressivity', 'visibility', 'ff_ratio']
X = data[features]
Y = data["label"]

# Encode the labels to be in the range [0, num_classes-1]
#label_encoder = LabelEncoder()
#Y_encoded = label_encoder.fit_transform(Y)

# Normalize data
scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
X_sc = scaler.fit_transform(X)

# Remove visibility & aggressivity from training
X_plot = X_sc[["visibility", "aggressivity"]].copy()
X_sc_removed = X_sc.drop(["visibility", "aggressivity"], axis=1)

# Slice and correct label
X_train = X_sc_removed[Y != 0]
Y_train = Y[Y != 0]

# Remap labels 2 to 1
Y_train[Y_train == -1] = 0

# Select a subset for training if needed
Y_train = Y_train[:50000]
X_train = X_train.iloc[:50000]

# Compter le nombre d'instances pour chaque label dans l'échantillon d'entraînement
label_counts_train = Y_train.value_counts()

# Afficher le nombre d'individus labellisés 1 et 0 dans l'échantillon d'entraînement
print("Nombre d'individus labellisés 1 dans l'échantillon d'entraînement :", label_counts_train[1])
print("Nombre d'individus labellisés 0 dans l'échantillon d'entraînement :", label_counts_train[0])

# Check for the number of instances of class 2 (now class 1)
num_twos = (Y_train == 1).sum()
print(f"Number of instances with label 1: {num_twos}")

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.30, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train.values)
x_test = torch.FloatTensor(X_test.values)
Y_train = torch.LongTensor(Y_train.values)
y_test = torch.LongTensor(Y_test.values)

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
hidden_sizes = [8, 4]
output_classes = 2 
model = Model(input_size, hidden_sizes, output_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
n_epochs = 15
batch_size = 256
losses, accuracies_train, accuracies_test = train_model(X_train, Y_train, x_test, y_test, model, criterion, optimizer, n_epochs, batch_size)

# Plot loss and accuracies
plt.plot(np.arange(1, n_epochs+1), losses, label='Loss')
plt.plot(np.arange(1, n_epochs+1), accuracies_train, label='Train Accuracy')
plt.plot(np.arange(1, n_epochs+1), accuracies_test, label='Test Accuracy')
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Charger les données de user_db_V1 depuis la base de données MongoDB
collec_v1 = db.user_db
data_v1 = pd.DataFrame(list(collec_v1.find()))

# Assurez-vous que les noms des caractéristiques correspondent
X_v1 = data_v1[features]

# Réappliquer le scaler avec les mêmes paramètres qu'il avait lors de l'entraînement
X_v1_sc = scaler.fit_transform(X_v1)

# Convertir les données en un DataFrame Pandas
X_v1_sc_df = pd.DataFrame(X_v1_sc, columns=X_v1.columns)

# Supprimer les colonnes "visibility" et "Aggressivity"
X_v1_removed = X_v1_sc_df.drop(["visibility", "aggressivity"], axis=1)

# Passer les données transformées dans le modèle entraîné pour obtenir les prédictions
X_v1_tensor = torch.FloatTensor(X_v1_removed.values)
print("Shape of X_v1_tensor:", X_v1_tensor.shape)


# Passer les données transformées dans le modèle entraîné pour obtenir les prédictions
model.eval()
with torch.no_grad():
    predictions = model(X_v1_tensor)
    _, predicted_labels = torch.max(predictions, 1)


plt.figure(figsize=(10, 6))
plt.scatter(X_v1_sc_df['visibility'], X_v1_sc_df['aggressivity'], c=predicted_labels, cmap='viridis', alpha=0.5)
plt.colorbar(ticks=[0, 1])
plt.xlabel('Visibility')
plt.ylabel('Aggressivity')
plt.title('Classification Results')
plt.show()

# Compter le nombre d'instances pour chaque label prédit
label_counts = pd.Series(predicted_labels.numpy()).value_counts()

# Ajouter les labels prédits aux données
X_v1_sc_df['predicted_label'] = predicted_labels.numpy()


# Calculer les statistiques descriptives pour chaque cluster
cluster_describe = X_v1_sc_df.groupby('predicted_label').describe()
print(cluster_describe)
