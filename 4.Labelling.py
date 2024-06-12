"""

Use visibility and aggressivity to label data
In the case  study of IF29 class
author : Nathan Davouse

"""


#Import librairies
from pymongo import MongoClient
import pandas as pd
import numpy as np

#connect from Mongo DB and import it on pandas
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_norm #whole database
data = pd.DataFrame(list(collec.find()))

#Define higher & lower bound of aggressivity and visibility
low_visi = np.percentile(data.visibility , 25)
high_visi = np.percentile(data.visibility , 85)
low_agr = np.percentile(data.aggressivity , 25)
high_agr = np.percentile(data.aggressivity , 85)

#functions to split 3 classes of visibility and aggressivity
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
# Split the data in 3
data["classes"] = np.round(0.5*(data.visibility.apply(split_visi)+data.aggressivity.apply(split_aggr)))

#pick sample
import random
random.seed(10) #for reproductibility
high_indexes = random.sample(list(data[data["classes"]==1].index),10000)
low_indexes = random.sample(list(data[data["classes"]==-1].index),10000)
#Give label to data
data["label"] = 0
data.loc[high_indexes,"label"] = 1
data.loc[low_indexes,"label"] = -1

#Export the collection to mongo
db.user_label.drop()
db.user_label.insert_many(data.drop("classes",axis=1).to_dict('records'))

#Plot 
import matplotlib.pyplot as plt
#plt.scatter(data.visibility[data["label"]==0],data.aggressivity[data["label"]==0],s=0.5,c="grey",label = "undefined_final")
plt.scatter(data.visibility[data["label"]==1],data.aggressivity[data["label"]==1],s=0.5,c="red",label = "suspicious")
#plt.scatter(data.visibility[data["classes"]==0],data.aggressivity[data["classes"]==0],s=0.5,c="black",label = "undefined_temp")
plt.scatter(data.visibility[data["label"]==-1],data.aggressivity[data["label"]==-1],s=0.5,c="blue",label = "non suspicious")

plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.show()

