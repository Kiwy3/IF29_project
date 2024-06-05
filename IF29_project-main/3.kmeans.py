"""

Use user database to perform a kmeans clustering
In the case  study of IF29 class
author : Nathan Davouse

"""

#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_V1 #whole database
#collec = db.user_db_sample #small db with 100 tweets
data = pd.DataFrame(list(collec.find()))
max_date = data.created_date.max()
#data["diff"] = data.created_date.apply(lambda x : (x - max_date).seconds)
data["diff"] = data.created_date.apply(lambda x: (x - max_date).seconds if x is not None else 0)

#choose feature
features = ["verified", "friend_nb", "listed_nb", "follower_nb", "favorites_nb","url_bool","len_description","tweet_nb","hash_avg","at_avg","tweet_user_count","diff"]
X = data[features]
debug = ["url_bool","retweet_avg"] #attribute to debug on file 2.user_db.py

#normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3) 
pca_data = pca.fit_transform(X_sc)
data["PCA_1"] = pca_data[:,0]
data["PCA_2"] = pca_data[:,1]
data["PCA_3"] = pca_data[:,2]

#Perform K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 100)
kmeans.fit(X_sc)
data["labels"] = kmeans.labels_


#plot
import matplotlib.pyplot as plt
plt.scatter(data[data.labels == 0].PCA_1,data[data.labels == 0].PCA_2,c="red",s=0.2)
plt.scatter(data[data.labels == 1].PCA_1,data[data.labels == 1].PCA_2,c="blue",s=0.2)
plt.show()

#3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[data.labels == 0].PCA_1, data[data.labels == 0].PCA_2, data[data.labels == 0].PCA_3)
ax.scatter(data[data.labels == 1].PCA_1, data[data.labels == 1].PCA_2, data[data.labels == 1].PCA_3)
plt.show()

#Export
db.create_collection("user_kmeans_pca")
db.user_kmeans_pca.drop()
db.user_kmeans_pca.insert_many(data.to_dict('records'))