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
sum_component = np.sum(np.abs(Contribution),axis=1)
Percent = np.divide(np.abs(Contribution.T),sum_component)*100

#Keep usefuls components and add _id
n_component = 9 #number of component to keep
X_save = X.iloc[:,:n_component]
X_save.insert(0,"_id",id_list)

#Export the collection to mongo
db.user_db_pca.drop()
db.user_db_pca.insert_many(X_save.to_dict('records'))

if plotting :
    # Visualisation des deux premières composantes principales
    plt.figure(figsize=(10, 7))
    sns.scatterplot(X,x='pca0', y='pca1')
    plt.title('ACP: Projection des deux premières composantes principales')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.savefig("./images/3_3.projection_2_composantes.png")
    plt.show()

    # Visualisation de la variance expliquée
    fig,ax1 = plt.subplots()
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',label="variance expliquée par la composante")
    ax2 = ax1.twinx()
    ax2.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid',label = "variance cumulée")
    ax1.set_ylabel('Ratio de variance expliquée')
    ax2.set_ylabel("Ratio de variance cumulée")
    ax1.set_xlabel("Nombre de composantes principales")
    ax1.vlines(9,0,0.175,colors="red")
    fig.suptitle('Variance expliquée par les composantes principales')
    fig.savefig("./images/3_3.pca_variance_cumsum.png")
    plt.show()

    #Participation au 2 premières composantes
    for n in Percent.columns[:2]:
        sort_indice = np.argsort(Percent[n])
        plt.barh(Percent.index[sort_indice],Percent[n][sort_indice])
        plt.title("Contribution à la composante "+n)
        plt.xlabel("pourcentage de contribution")
        plt.tight_layout()
        plt.savefig("./images/3_3.contribution_"+n+".png")
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


#close the mongodb connection
client.close()
