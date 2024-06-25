import pandas as pd
from pymongo import MongoClient

# Connexion à MongoDB et importation des données dans Pandas
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db
data = pd.DataFrame(list(collec.find()))

# Colonnes à analyser
features_to_analyse = [
    "friend_nb", "listed_nb", "follower_nb", "favorites_nb", "len_description",
    "tweet_nb", "tweet_user_count", "user_lifetime", "aggressivity", "visibility", "ff_ratio"
]

# Liste des colonnes pour le groupement
features_to_group = ["svm", "nn", 'cah']  # Remplacez par vos propres colonnes

# Création du fichier Excel avec plusieurs onglets
output_file = "summary_statistics_multi.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for feature in features_to_group:
        # Calculer la moyenne et la médiane pour chaque colonne, groupées par la feature actuelle
        mean_df = data.groupby(feature)[features_to_analyse].mean().reset_index()
        median_df = data.groupby(feature)[features_to_analyse].median().reset_index()

        # Renommer les colonnes pour indiquer qu'elles sont des moyennes et des médianes
        mean_df = mean_df.rename(columns={col: col + '_mean' for col in features_to_analyse})
        median_df = median_df.rename(columns={col: col + '_median' for col in features_to_analyse})

        # Fusionner les DataFrames de moyenne et de médiane
        summary_df = pd.merge(mean_df, median_df, on=feature)

        # Écrire les résultats dans un onglet du fichier Excel
        summary_df.to_excel(writer, sheet_name=feature, index=False)

print(f"Le fichier Excel '{output_file}' a été créé avec succès.")
