"""

Make user database from tweet database with mongo aggregate
In the case  study of IF29 class
author : Nathan Davouse

"""
#Import librairies
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Connection with mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db

data = pd.DataFrame(list(collec.find()))
id_list = data.pop("_id")

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
