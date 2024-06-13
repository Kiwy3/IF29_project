import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pymongo import MongoClient

# Connexion à MongoDB et chargement des données
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db
data = pd.DataFrame(list(collec.find()))

data = data.sample(frac=0.002)
print(data.shape)

# Mise à l'échelle des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['aggressivity', 'visibility']])

# Application de DBSCAN
# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=200)  # You may need to tune the parameters
labels = dbscan.fit_predict(data_scaled)
data['label'] = labels

# Imprimer le nombre d'individus pour chaque label dans les données équilibrées
print(data['label'].value_counts())

# Exporter les résultats vers MongoDB
db.drop_collection("user_label_DBSCAN")
db.create_collection("user_label_DBSCAN")
db.user_label_DBSCAN.insert_many(data.to_dict('records'))


# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Couleur pour le bruit (cluster -1)
    class_member_mask = (labels == k)
    xy = data_scaled[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
plt.title('Clusters par DBSCAN')
plt.xlabel('Aggressivity (scaled)')
plt.ylabel('Visibility (scaled)')
plt.show()
