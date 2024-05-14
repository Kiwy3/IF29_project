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
#collec = db.user_db_V1 #whole database
collec = db.user_db_sample #small db with 100 tweets
data = pd.DataFrame(list(collec.find()))
time_2 = time.localtime()

#choose feature
features = list(data.columns)
del features[0:3] #to remove id, name and created_date
features = ["hash_avg", "friend_nb", "listed_nb", "follower_nb", "favorites_nb", "tweet_nb", "tweet_nb"]
X = data[features]


#normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
T = pd.DataFrame(X_sc)

#Perform K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
kmeans.fit(X_sc)
data["labels"] = kmeans.labels_

for i in range(len(X)):
    collec.update_one(
    {"_id" : data.loc[i,"_id"].value() },
    {"$set" : {"kmeans_label" : data.loc[i,"labels"].value()}}
)

#Add label to mongodb
"""
docs = collec.update_many(
    {"_id" : data["_id"] },
    {"$set" : {"kmeans_label" : data["labels"]}}
)"""