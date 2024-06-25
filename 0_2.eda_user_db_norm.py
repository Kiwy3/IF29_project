import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier CSV dans un DataFrame pandas
user_db = pd.read_csv('Old/user_db_norm.csv')
 
# Afficher les premières lignes du DataFrame pour comprendre la structure des données
print(user_db.head())

# Vérifier les informations sur les données et les types de colonnes
print(user_db.info())

# Afficher les statistiques descriptives pour les variables numériques
print(user_db.describe())

# Vérifier les valeurs manquantes
print(user_db.isnull().sum())

# Tracer un histogramme pour les variables numériques
user_db.hist(figsize=(12, 10))
plt.show()

# Tracer des graphiques à barres pour les variables catégorielles
# Par exemple, si verified est une variable catégorielle
plt.figure(figsize=(8, 6))
sns.countplot(x='verified', data=user_db)
plt.title('Répartition des Comptes Vérifiés')
plt.show()

# Tracer des diagrammes de dispersion pour explorer les relations
# Par exemple, si vous voulez voir la relation entre follower_nb et listed_nb
plt.figure(figsize=(8, 6))
sns.scatterplot(x='follower_nb', y='listed_nb', data=user_db)
plt.title('Relation entre Followers et Listes')
plt.show()

# Calculer la matrice de corrélation
correlation_matrix = user_db.corr()

# Tracer une heatmap pour visualiser la corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show() 
