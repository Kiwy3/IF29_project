"""

Use visibility and aggressivity to label data
In the case  study of IF29 class
author : Nathan Davouse

"""


#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_V2 #whole database
#collec = db.user_db_sample #small db with 100 tweets
data = pd.DataFrame(list(collec.find()))

#Scale the features
features = ["visibility","Aggressivity"]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
X = scaler.fit_transform(data[features])

#Split in 3 classes : 
X["total"] = X["visibility"] + X["Aggressivity"]
X.sort_values("total",inplace=True)
X["label"] = 1
X.iloc[:10000,3]=0
X.iloc[-10000:,3]=2

#Put the index back in the data field
X.sort_index(inplace=True,ascending=True)
data["label"] = X["label"]

#Export the collection to mongo
db.user_label.drop()
db.user_label.insert_many(data.to_dict('records'))

#Plot 
"""
import matplotlib.pyplot as plt
plt.scatter(X.visibility[X["label"]==0],X.Aggressivity[X["label"]==0],s=0.5,c="blue",label = "non suspicious")
plt.scatter(X.visibility[X["label"]==1],X.Aggressivity[X["label"]==1],s=0.5,c="grey",label = "undefined")
plt.scatter(X.visibility[X["label"]==2],X.Aggressivity[X["label"]==2],s=0.5,c="red",label = "suspicious")

plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.show()"""
