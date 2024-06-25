from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
import pandas as pd

# Connexion à MongoDB et importation des données dans Pandas
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_pca
data = pd.DataFrame(list(collec.find()))

# Sauvegarder les IDs avant de retirer la colonne _id
id_list = data["_id"]
data = data.drop(columns=["_id"])

# Paramètres du KMeans
n_split = 10
k = 3

# Appliquer KMeans
X = data.loc[:, "pca0":"pca8"]
kmeans = MiniBatchKMeans(n_clusters=k, verbose=1, random_state=42, max_no_improvement=100)
kmeans.set_output(transform="pandas")
Projection = kmeans.fit_transform(X)
data["partition"] = kmeans.labels_
Centroids = pd.DataFrame(kmeans.cluster_centers_)

# Plot KMeans results
plt.scatter(X["pca0"], X["pca1"], c=data.partition, s=0.5)
plt.scatter(Centroids.loc[:, 0], Centroids.loc[:, 1], c="red", label="centroids")
plt.title("Clusters centroids and repartition")
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.xlim(-5, 50)
plt.ylim(-20, 20)
plt.legend()
plt.savefig("./images/4_1.minikmeans_representation.png")
plt.show()

# Plot KMeans results on 3D
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X["pca0"], X["pca1"], X["pca2"], c=data.partition, s=0.5)
ax.scatter(Centroids.loc[:, 0], Centroids.loc[:, 1], Centroids.loc[:, 2], c="red", label="centroids")
plt.suptitle("Clusters centroids and repartition")
ax.set_xlabel('Première composante principale')
ax.set_ylabel('Deuxième composante principale')
ax.set_xlim(-5, 50)
ax.set_ylim(-20, 20)
ax.legend()
plt.savefig("./images/4_1.3D_minikmeans_representation.png")
plt.show()

# Charger les données de la base de données B
db_B = client['IF29']
collection_B = db_B['user_db']

# Réintroduire les IDs dans le DataFrame
data["_id"] = id_list

# Mettre à jour les documents de la base de données B avec les étiquettes de partition
for index, row in data.iterrows():
    id_value = row['_id']
    cluster_label = int(row['partition'])  # Assurez-vous que le label est converti en int si nécessaire
    collection_B.update_one({'_id': id_value}, {'$set': {'kmeans': cluster_label}})

print("Mise à jour des étiquettes de partition terminée.")
