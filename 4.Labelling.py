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

low_visi = X.visibility.describe()["25%"]
high_visi = X.visibility.describe()["75%"]
low_agr = X.Aggressivity.describe()["25%"]
high_agr = X.Aggressivity.describe()["75%"]

def split_aggr(x):
    if x<low_agr:
        return -1
    elif x> high_agr:
        return 1
    else  : return 0

def split_visi(x):
    if x<low_visi:
        return -1
    elif x> high_visi:
        return 1
    else  : return 0

X["classes"] = X.visibility.apply(split_visi)*X.Aggressivity.apply(split_visi)

#pick sample
import random
random.seed(10) #for reproductibility
high_indexes = random.sample(list(X[X["classes"]==1].index),10000)
low_indexes = random.sample(list(X[X["classes"]==-1].index),10000)
#Give label to data
X["label"] = 0
X.loc[high_indexes,"label"] = 1
X.loc[low_indexes,"label"] = -1

data["label"] = X["label"]

#Export the collection to mongo
#db.user_label.drop()
#db.user_label.insert_many(data.to_dict('records'))

#Plot 
import matplotlib.pyplot as plt
plt.scatter(X.visibility[X["label"]==-1],X.Aggressivity[X["label"]==-1],s=0.5,c="blue",label = "non suspicious")
plt.scatter(X.visibility[X["label"]==0],X.Aggressivity[X["label"]==0],s=0.5,c="grey",label = "undefined")
plt.scatter(X.visibility[X["label"]==1],X.Aggressivity[X["label"]==1],s=0.5,c="red",label = "suspicious")
plt.scatter(X.visibility[X["classes"]==0],X.Aggressivity[X["classes"]==0],s=0.5,c="black",label = "undefined")


plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.show()
