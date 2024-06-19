"""

Use user database to perform a kmeans clustering
In the case  study of IF29 class
author : Nathan Davouse

"""
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_pca #whole database
data = pd.DataFrame(list(collec.find()))
id_list = data.pop("_id")
X = data
# Calculer les scores de silhouette pour k variant de 2 à 10
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42,verbose=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_scores.append(metrics.silhouette_score(X, labels))
    print("S_score : ",k,"/9")
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