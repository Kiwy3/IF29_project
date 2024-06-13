import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from pymongo import MongoClient

# Connexion à MongoDB et chargement des données
client = MongoClient("localhost", 27017)
db = client.IF29
collec = db.user_db_norm
data_db = pd.DataFrame(list(collec.find()))
data = data_db.sample(frac=0.02, random_state=42)  # Pour reproductibilité

features = [
    'verified', 'protected', 'friend_nb', 'listed_nb', 'follower_nb',
    'favorites_nb', 'len_description', 'hash_avg', 'mention_avg', 'url_avg',
    'symbols_avg', 'tweet_nb', 'tweet_user_count', 'user_lifetime', 
    'tweet_frequency', 'friend_frequency', 'aggressivity', 'visibility', 'ff_ratio'
]

# Sélection des features
X = data[features]

# Analyse en Composantes Principales (ACP)
pca = PCA()
pca_res = pca.fit_transform(X)

# Choix automatique du nombre de composantes
threshold = 0.95  # seuil de variance expliquée cumulative à atteindre
num_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1

print(f'Number of components to retain {threshold*100}% variance: {num_components}')

# Réduction des dimensions de pca_res
pca_res_reduced = pca_res[:, :num_components]

# Optimisation dynamique du nombre de clusters avec AgglomerativeClustering
silhouette_scores = []
for n_clusters in range(2, 11):  # Testez différentes valeurs de n_clusters
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = agg_cluster.fit_predict(pca_res_reduced)
    silhouette_avg = silhouette_score(pca_res_reduced, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre optimal de clusters pour AgglomerativeClustering
optimal_n_clusters_agg = np.argmax(silhouette_scores) + 2  # +2 car nous avons commencé à n_clusters = 2

print(f'Optimal number of clusters for Agglomerative Clustering based on silhouette score: {optimal_n_clusters_agg}')

# Appliquer AgglomerativeClustering avec le nombre optimal de clusters
agg_cluster = AgglomerativeClustering(n_clusters=optimal_n_clusters_agg, linkage='ward')
cluster_labels_agg = agg_cluster.fit_predict(pca_res_reduced)

# Calcul du coefficient de silhouette pour K-means
silhouette_scores_kmeans = []
for n_clusters in range(2, 11):  # Testez différentes valeurs de n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pca_res_reduced)
    kmeans_labels = kmeans.labels_
    silhouette_avg = silhouette_score(pca_res_reduced, kmeans_labels)
    silhouette_scores_kmeans.append(silhouette_avg)

# Trouver le nombre optimal de clusters pour K-means
optimal_n_clusters_kmeans = np.argmax(silhouette_scores_kmeans) + 2  # +2 car nous avons commencé à n_clusters = 2

print(f'Optimal number of clusters for K-means based on silhouette score: {optimal_n_clusters_kmeans}')

# Appliquer K-means avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_n_clusters_kmeans, random_state=42)
kmeans.fit(pca_res_reduced)
kmeans_labels = kmeans.labels_

# Tracé du dendrogramme
plt.figure(figsize=(12, 8))
dendrogram(linkage(pca_res_reduced, method='ward'), truncate_mode='level', p=5)
plt.title('Dendrogramme de la Classification Ascendante Hiérarchique (CAH)')
plt.xlabel('Exemples')
plt.ylabel('Distance euclidienne')
plt.show()

# Visualisation des clusters après CAH
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_res_reduced[:, 0], y=pca_res_reduced[:, 1], hue=cluster_labels_agg, palette='viridis')
plt.title(f'Répartition des clusters après Classification Ascendante Hiérarchique (CAH, {optimal_n_clusters_agg} clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Visualisation des clusters après K-means
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_res_reduced[:, 0], y=pca_res_reduced[:, 1], hue=kmeans_labels, palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
plt.title(f'Répartition des clusters après K-means (PCA-reduced, {optimal_n_clusters_kmeans} clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
