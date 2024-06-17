#Import librairies
from pymongo import MongoClient
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Connection with mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_norm
data = pd.DataFrame(list(collec.find()))
id_list = data.pop("_id")

#Variable de controle
plotting = False



# Calcul des valeurs propres et des vecteurs propres via PCA
pca = PCA(12)
pca.set_output(transform="pandas")
X = pca.fit_transform(data)

#Variance et attributes
explained_variance = pca.explained_variance_ratio_
projection = pca.components_




if plotting :
    # Visualisation des deux premières composantes principales
    plt.figure(figsize=(10, 7))
    sns.scatterplot(X,x='pca0', y='pca1')
    plt.title('ACP: Projection des deux premières composantes principales')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.show()

    # Visualisation de la variance expliquée
    plt.figure(figsize=(10, 7))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center')
    plt.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid')
    plt.ylabel('Ratio de variance expliquée')
    plt.xlabel('Composantes principales')
    plt.title('Variance expliquée par les composantes principales')
    plt.show()