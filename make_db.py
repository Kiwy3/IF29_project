import pymongo
from pymongo import MongoClient
import jsonlines
import os

#connect from mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.wc_import

folder_path = "C:\\Users\\Nathan\\IF29_project\\raw"
total = len(os.listdir(folder_path))
i=0
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        with jsonlines.open(os.path.join(folder_path, filename)) as f:
            for line in f:
                collec.insert_one(line)
            i=i+1
            print("file "+ str(filename)+" imported // ",i ,"/",total)
              
client.close()