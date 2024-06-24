"""

Use visibility and aggressivity to label data
In the case  study of IF29 class
author : Nathan Davouse

"""


#Import librairies
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#connect from Mongo DB and import it on pandas
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_norm #whole database
data = pd.DataFrame(list(collec.find()))


#Define higher & lower bound of aggressivity and visibility
low_visi = np.percentile(data.visibility , 50)
high_visi = np.percentile(data.visibility , 85)
low_agr = np.percentile(data.aggressivity , 50)
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
n_samples = 15000
random.seed(10) #for reproductibility
high_indexes = random.sample(list(data[data["classes"]==1].index),n_samples)
low_indexes = random.sample(list(data[data["classes"]==-1].index),n_samples)
#Give label to data
data["label"] = 0
data.loc[high_indexes,"label"] = 1
data.loc[low_indexes,"label"] = -1

#Export the collection to mongo
db.user_label.drop()
db.user_label.insert_many(data.drop("classes",axis=1).to_dict('records'))

#plot aggressivity and visibility hist
fig,axs = plt.subplots(2)
fig.suptitle("Distribution de la visibilité \n et de l'aggressivité",x=0.4)
visi_hist = axs[0].hist(data.visibility,range = (-2,5),bins = 25,density = True)
higher_bound_visi = max(visi_hist[0])
axs[0].vlines(low_visi,0,higher_bound_visi,colors="red")
axs[0].vlines(high_visi,0,higher_bound_visi,colors="red")
axs[0].set_ylabel("visibilité")

agr_hist = axs[1].hist(data.aggressivity,range = (-1,2),bins = 25,density=True)
higher_bound_agre = max(agr_hist[0])
axs[1].vlines(low_agr,0,higher_bound_agre,colors="red",label = "Bornes de \n labellisation")
axs[1].vlines(high_agr,0,higher_bound_agre,colors="red")
axs[1].set_ylabel("aggressivité")
axs[1].set_xlabel("Densité de probabilité dans l'intervalle")
fig.legend(loc='upper right')
fig.savefig("./images/5_1.aggr_visi_hist.png")
plt.show()


#Plot 
import matplotlib.pyplot as plt
plt.scatter(data.visibility[data["label"]==1],data.aggressivity[data["label"]==1],s=0.5,c="red",label = "suspicious")
plt.scatter(data.visibility[data["label"]==-1],data.aggressivity[data["label"]==-1],s=0.5,c="blue",label = "non suspicious")
plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.title("")
plt.savefig("./images/5_1.labelling_results.png")
plt.show()

