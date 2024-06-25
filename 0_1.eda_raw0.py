import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from textblob import TextBlob

# Chargement des données JSON
data = []
with open('Old/raw0.json', encoding="utf8") as f:
    x = f.readlines()

for i in range(len(x)):
    data.append(json.loads(x[i]))

# Normalisation des données JSON
data = pd.json_normalize(data)

# Comptage des valeurs NaN par colonne
col = data.columns
Nacount = [data[i].isna().sum() for i in col]

# Création d'un DataFrame pour les attributs et le comptage des NaN
Stud = pd.DataFrame(col, columns=["Attribut"])
total_rows = data.shape[0]
NA_lim = total_rows * 0.8
Stud["NA_count"] = Nacount
Stud["NA_filter"] = Stud["NA_count"].apply(lambda x: 1 if x > NA_lim else 0)

# Filtrage des attributs liés à l'utilisateur
Stud["user_filter"] = Stud["Attribut"].apply(lambda x: 1 if x.startswith("user.") else 0)
user_attributes = Stud[Stud["user_filter"] == 1].drop("NA_filter", axis=1)

# Analyse des niveaux et de l'origine des attributs
Stud["lvl"] = Stud["Attribut"].apply(lambda x: x.count("."))
Stud["origine"] = Stud["Attribut"].apply(lambda x: x.split(".")[0] if x.count(".") > 0 else "/")

# Filtrage des colonnes à supprimer
cols_to_drop = Stud[Stud["NA_filter"] == 1]["Attribut"].tolist()

# Suppression des colonnes de data
data.drop(columns=cols_to_drop, inplace=True)

# Suppression des attributs correspondants de Stud
Stud = Stud[Stud["NA_filter"] != 1]

# Visualisation des valeurs manquantes par colonne
plt.figure(figsize=(10, 6))
sns.barplot(x=Stud["Attribut"], y=Stud["NA_count"])
plt.xticks(rotation=90)
plt.title("Nombre de valeurs manquantes par colonne")
plt.show()

# Distribution des niveaux des attributs
plt.figure(figsize=(10, 6))
sns.countplot(x=Stud["lvl"])
plt.title("Distribution des niveaux des attributs")
plt.show()

# Distribution des origines des attributs
plt.figure(figsize=(10, 6))
sns.countplot(x=Stud["origine"])
plt.xticks(rotation=90)
plt.title("Distribution des origines des attributs")
plt.show()

# Visualisation des attributs utilisateur
print(user_attributes)

# Analyse de la longueur des tweets (si les tweets sont présents dans les données)
data['tweet_length'] = data['retweeted_status.extended_tweet.full_text'].str.len()
plt.figure(figsize=(10, 6))
sns.histplot(data['tweet_length'], kde=True)
plt.title("Distribution de la longueur des tweets")
plt.show()

# Affichage des mots les plus fréquents dans les tweets

def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Suppression des mentions
    text = re.sub(r'#', '', text)               # Suppression des hashtags
    text = re.sub(r'http\S+', '', text)         # Suppression des URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)     # Suppression des caractères non alphabétiques
    return text.lower()

# Suppression des lignes avec des valeurs NaN dans 'retweeted_status.extended_tweet.full_text'
data = data.dropna(subset=['retweeted_status.extended_tweet.full_text'])

# Application de la fonction de prétraitement aux tweets
data['cleaned_tweet'] = data['retweeted_status.extended_tweet.full_text'].apply(preprocess_text)

# Combinaison de tous les tweets en une seule chaîne de caractères
all_words = ' '.join(data['cleaned_tweet'])

# Comptage de la fréquence des mots
word_freq = Counter(all_words.split())

# Affichage des 30 mots les plus fréquents
common_words = word_freq.most_common(30)
print(common_words)

# Visualisation avec un nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Afficher un résumé des données
print(data.describe(include='all'))

# Comptage des tweets par utilisateur
tweets_per_user = data['user.id_str'].value_counts()

# Afficher les 20 utilisateurs les plus actifs
top_users = 20
top_tweeters = tweets_per_user.head(top_users)

# Graphique des utilisateurs les plus actifs
plt.figure(figsize=(12, 8))
top_tweeters.plot(kind='bar', color='salmon')
plt.title(f'Top {top_users} Utilisateurs par Nombre de Tweets')
plt.xlabel('ID Utilisateur')
plt.ylabel('Nombre de Tweets')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

# Extraire les hashtags et les compter
hashtags = data['entities.hashtags'].apply(lambda x: [tag['text'] for tag in x] if x else [])
hashtags_flat = [item for sublist in hashtags for item in sublist]
hashtag_counts = pd.Series(hashtags_flat).value_counts()

# Afficher les 20 hashtags les plus fréquents
top_n = 20
top_hashtag_counts = hashtag_counts.head(top_n)

# Nuage de mots des hashtags
text = ' '.join(hashtags_flat)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des Hashtags')
plt.show()

# Graphique de la fréquence des hashtags
plt.figure(figsize=(12, 8))
top_hashtag_counts.plot(kind='bar', color='skyblue')
plt.title(f'Top {top_n} Hashtags par Fréquence')
plt.xlabel('Hashtags')
plt.ylabel('Fréquence')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

# Répartition des sources des tweets
source_counts = data['source'].value_counts().head(10)  # Top 10 sources
print(source_counts)

# Graphique des top sources des tweets
plt.figure(figsize=(12, 8))
sns.barplot(y=source_counts.index, x=source_counts.values, palette='viridis')
plt.title('Top 10 Sources des Tweets')
plt.xlabel('Nombre de Tweets')
plt.ylabel('Source')
plt.show()

# Types de médias partagés
media_types = data['retweeted_status.extended_tweet.extended_entities.media'].apply(lambda x: [media['type'] for media in x] if isinstance(x, list) else [])
media_types_flat = [item for sublist in media_types for item in sublist]
media_type_counts = pd.Series(media_types_flat).value_counts()
print(media_type_counts)

# Graphique des types de médias partagés
plt.figure(figsize=(12, 8))
sns.barplot(y=media_type_counts.index, x=media_type_counts.values, palette='viridis')
plt.title('Répartition des Types de Médias Partagés')
plt.xlabel('Nombre de Médias')
plt.ylabel('Type de Médias')
plt.show()

# Fonction pour analyser les sentiments
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Analyse des sentiments des tweets
data['sentiment'] = data['retweeted_status.text'].apply(lambda x: get_sentiment(x) if isinstance(x, str) else get_sentiment(data['text'][0]))
data[['polarity', 'subjectivity']] = pd.DataFrame(data['sentiment'].tolist(), index=data.index)

# Visualisation des sentiments
plt.figure(figsize=(10, 6))
sns.histplot(data['polarity'], bins=30, kde=True)
plt.title('Distribution des Polarités des Sentiments')
plt.xlabel('Polarité')
plt.ylabel('Fréquence')
plt.show()