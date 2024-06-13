"""

Use user database to perform a kmeans clustering
In the case  study of IF29 class
author : Nathan Davouse

"""

#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt


client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db #whole database
#collec = db.user_db_sample #small db with 100 tweets
data = pd.DataFrame(list(collec.find()))
max_date = data.created_date.max()
data["diff"] = data.created_date.apply(lambda x : (x - max_date).seconds)


#choose feature
features = ["verified","protected","friend_nb","listed_nb","follower_nb","favorites_nb","len_description","hash_avg","mention_avg","url_avg","symbols_avg","tweet_nb","tweet_user_count","user_lifetime","tweet_frequency","friend_frequency","aggressivity","visibility","ff_ratio"]
X = data[features]
debug = ["url_bool","retweet_avg"] #attribute to debug on file 2.user_db.py

# Normalisation des données
#normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(X)

# Calculer les scores de silhouette pour k variant de 2 à 10
silhouette_scores = []
from sklearn.cluster import KMeans
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(data_normalized, labels))

# Afficher les scores de silhouette
print("Silhouette Scores for each k:", silhouette_scores)

# Tracer les scores de silhouette
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Silhouette Scores for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Implémenter la méthode du coude
def find_elbow_point(silhouette_scores):
    deltas = np.diff(silhouette_scores)
    delta_deltas = np.diff(deltas)
    return np.argmax(delta_deltas) + 2  # +2 pour compenser les différences successives

optimal_k = find_elbow_point(silhouette_scores)
print(f"Optimal number of clusters according to the elbow method: {optimal_k}")

# Appliquer KMeans avec le nombre optimal de clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(data_normalized)
labels_optimal = kmeans_optimal.labels_

# Ajouter les labels aux données
X['cluster'] = labels_optimal

# Afficher les données avec les clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['visibility'], X['aggressivity'], c=X['cluster'], cmap='viridis')
plt.title("Clusters of Users based on Visibility and Aggressivity")
plt.xlabel("Visibility")
plt.ylabel("Aggressivity")
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()