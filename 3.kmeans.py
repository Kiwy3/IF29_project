"""

Use user database to perform a kmeans clustering
In the case  study of IF29 class
author : Nathan Davouse

"""
#Time librairy to obtain the calculation time
import time
time_1 = time.localtime()
def time_diff(st, end):
    sec = end.tm_sec - st.tm_sec + 60*(end.tm_min - st.tm_min)
    sec_res = sec%60
    minu = (sec-sec_res)/60
    return minu, sec_res

def time_print(st,end):
    m,s = time_diff(st, end)
    print("temps de run : ",m,"minutes et ",s," secondes")


#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_V1 #whole database
#collec = db.user_db_sample #small db with 100 tweets
data = pd.DataFrame(list(collec.find()))
time_2 = time.localtime()

#choose feature
features = list(data.columns)
del features[0:3] #to remove id, name and created_date
features = ["verified", "friend_nb", "listed_nb", "follower_nb", "favorites_nb","url_bool","len_description","tweet_nb","hash_avg","at_avg","tweet_user_count"]
X = data[features]
debug = ["url_bool","retweet_avg"]


#normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2) 
pca_data = pca.fit_transform(X_sc)
data["PCA_1"] = pca_data[:,0]
data["PCA_2"] = pca_data[:,1]

#Perform K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 100)
kmeans.fit(X_sc)

#plot
import matplotlib.pyplot as plt
plt.scatter(data[data.labels == 0].PCA_1,data[data.labels == 0].PCA_2,c="red")
plt.scatter(data[data.labels == 1].PCA_1,data[data.labels == 1].PCA_2,c="blue")

#Export
data["labels"] = kmeans.labels_
export = data.loc[:,["labels","_id"]]
#db.KM.drop()
#db.KM.insert_many(export.to_dict('records'))