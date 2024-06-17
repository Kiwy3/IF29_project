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
plotting = True
cercle = True



# Calcul des valeurs propres et des vecteurs propres via PCA
pca = PCA(12)
pca.set_output(transform="pandas")
X = pca.fit_transform(data)

#Variance et attributes
explained_variance = pca.explained_variance_ratio_
Contribution = pd.DataFrame(pca.components_,columns=data.columns,index=X.columns)




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
    plt.savefig("./images/3_3.projection_2_composantes.png")
    plt.show()

if cercle:
    components = pca.components_.T

    plt.figure(figsize=(10, 10))
    plt.quiver(np.zeros(components.shape[0]), np.zeros(components.shape[0]), 
            components[:, 0], components[:, 1], 
            angles='xy', scale_units='xy', scale=1)

    for i, feature in enumerate(data.columns):
        plt.text(components[i, 0], components[i, 1], feature, color='blue', ha='center', va='center')

    # Délimiter le cercle unité
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--', lw=1)
    plt.axvline(0, color='grey', linestyle='--', lw=1)
    plt.xlabel('axe 1')
    plt.ylabel('axe 2')
    plt.title('Cercle des corrélations')
    plt.rcParams.update({'font.size': 14})
    plt.savefig("./images/3_3.correlation_circle.png")
    plt.show()