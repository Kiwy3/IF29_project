import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_label
data = pd.DataFrame(list(collec.find()))

# Define features and target variable
data_removed = data[data["label"] != 0]

# Define features and target variable
features = [
    "verified","protected","friend_nb","listed_nb","follower_nb","favorites_nb","len_description","hash_avg","mention_avg","url_avg","symbols_avg","tweet_nb","tweet_user_count","user_lifetime","tweet_frequency","friend_frequency","aggressivity","visibility","ff_ratio"
]
X = data_removed[features]
y = data_removed["label"]

y_loc = y.iloc[:200000]
X_loc = X.iloc[:200000]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_loc, y_loc, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel="linear")

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Préparer les données pour la prédiction
X_full = data[features]
X_full = scaler.transform(X_full)  # Normaliser les données

# Prédire les labels avec le modèle SVM
y_pred_full = svm_classifier.predict(X_full)

# Ajouter les labels prédits au DataFrame
data["predicted_label"] = y_pred_full

# Créer un graphique de visualisation
plt.figure(figsize=(12, 8))
scatter = plt.scatter(data["visibility"], data["Aggressivity"], c=data["predicted_label"], cmap='viridis', alpha=0.5)
plt.xlabel("Visibility")
plt.ylabel("Aggressivity")
plt.title("Predicted Labels based on Visibility and Aggressivity")
plt.legend(*scatter.legend_elements(), title="Predicted Label")
plt.grid(True)
plt.show()

