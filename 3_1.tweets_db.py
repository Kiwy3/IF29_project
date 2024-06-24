"""

Import many json file of tweets to only one mongo database
In the case  study of IF29 class
author : Nathan Davouse

"""

#Import librairies
from pymongo import MongoClient
import jsonlines
import os

#connect from mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.tweets_global

#Initialisation
folder_path = "C:\\Users\\Nathan\\IF29_project\\raw"
total = len(os.listdir(folder_path))
i=0
#loop on everyfile of the folder
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        #open current file with jsonlines
        with jsonlines.open(os.path.join(folder_path, filename)) as f:
            for line in f:
                collec.insert_one(line) #add one document on the collection
            #follow the number of file and print the current step  
            i=i+1
            print("file "+ str(filename)+" imported // ",i ,"/",total)

#Close the connection with mongodb             
client.close()   
