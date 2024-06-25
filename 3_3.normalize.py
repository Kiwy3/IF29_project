"""

Make user database from tweet database with mongo aggregate
In the case  study of IF29 class
author : Nathan Davouse

"""
#Import librairies
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Connection with mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db

data = pd.DataFrame(list(collec.find()))
id_list = data.pop("_id")

#Plot mean and std of attributes
mean_df = pd.DataFrame(data.mean(axis = 0))
std_df = pd.DataFrame(data.std(axis = 0))
fig, ax = plt.subplots(2)
ax[0].barh(mean_df.index,mean_df[0])
ax[1].barh(std_df.index,std_df[0])
fig.suptitle("Effect of the normalization")
ax[0].set_title("mean of the attribute")
ax[1].set_xlabel("standard deviation of the attribute")
fig.savefig("./images/3_3.mean_std_normalize.png")
plt.show()

features = ['verified', 'friend_nb',
        'listed_nb', 'follower_nb', 'favorites_nb', 'len_description',
        'hash_avg', 'mention_avg', 'url_avg', 'symbols_avg', 'tweet_nb',
        'tweet_user_count', 'user_lifetime', 'tweet_frequency',
        'friend_frequency', 'aggressivity', 'visibility', 'ff_ratio']


#Scale the features
scaler = StandardScaler()
scaler.set_output(transform="pandas")
X = scaler.fit_transform(data)

X["_id"] = id_list

#Export the collection to mongo
db.user_db_norm.drop()
db.user_db_norm.insert_many(X.to_dict('records'))

#close the mongodb connection
client.close()
