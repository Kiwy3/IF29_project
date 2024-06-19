"""

Use user database to perform a kmeans clustering
In the case  study of IF29 class
author : Nathan Davouse

"""
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import time

#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_pca #whole database
data = pd.DataFrame(list(collec.find()))
id_list = data.pop("_id")
n_10th = int(len(data)/10)
X = data.sample(n_10th,random_state=42)
# Calculer les scores de silhouette pour k variant de 2 à 10
silhouette_scores = []
inertia_score = []
centroids = []
k_list = [2,3,4,5,6,7]

def time_diff(st, end):
    sec = end.tm_sec - st.tm_sec + 60*(end.tm_min - st.tm_min)
    sec_res = sec%60
    minu = (sec-sec_res)/60
    print("temps : ",minu,"minutes et ",sec_res," secondes")

for k in k_list:
    print("\n Kmeans for "+str(k)+" clusters : ")
    start_time = time.localtime()
    kmeans = KMeans(n_clusters=k, random_state=42,verbose=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    inertia_score.append(kmeans.inertia_)
    centroids.append(kmeans.cluster_centers_)
    silhouette_scores.append(metrics.silhouette_score(X, labels))
    time_diff(start_time,time.localtime())
# Afficher les scores de silhouette
print("Silhouette Scores for each k:", silhouette_scores)

plt.scatter(X["pca0"],X["pca1"],c = labels)
plt.xlim(0,50)
plt.ylim(-20,20)
plt.show()

# Tracer les scores de silhouette
plt.figure(figsize=(10, 6))
plt.plot(k_list, silhouette_scores, marker='o')
plt.title("Silhouette Scores for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("./images/4_1.silhouette_score.png")
plt.show()

# Tracer les scores de silhouette
plt.figure(figsize=(10, 6))
plt.plot(k_list, inertia_score, marker='o')
plt.title("Inertia for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("./images/4_1.inertia.png")
plt.show()