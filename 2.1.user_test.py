"""

Make user database from tweet database with mongo aggregate
In the case  study of IF29 class
author : Nathan Davouse

"""
#Import librairies
from pymongo import MongoClient
import time

#Connection with mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.tweets_sample
from datetime import datetime
max_date = datetime(2018, 7, 1)


#Make the aggregation pipeline
pipeline = [
    {"$sort":{"current_time":1}}, #to allow $last to effectively be the last
    {"$limit": 10}, #if we need to test it on few lines
    {"$addFields" : {"user_date" : {"$dateFromString":{"dateString" : "$user.created_at"}}  }},
    #group by user 
    {"$group":{ #user global information
               "_id":"$user.id",
               "name" : {"$last":"$user.name"},
               "created_date" : {"$last":"$user_date"},
               #user numeric information
               "friend_nb":{"$avg":"$user.friends_count"},
               "listed_nb":{"$avg":"$user.listed_count"},
               "follower_nb":{"$avg":"$user.followers_count"},
               #tweet stats
               "tweet_user_count":{"$max":"$user.statuses_count"} 
               }},
    {"$addFields": { "total_life_days": 
                    {"$dateDiff": {"startDate" : "$created_date",
                                   "endDate" : max_date,
                                   "unit" : "day"}}
    }}

    #Export it on another database
    ,{"$out" : "test"}
]

st = time.localtime() #to collect the time of start
collec.aggregate(pipeline)
end = time.localtime() #to collect the time of end of the aggregation

#print the doc for test
"""test = collec.aggregate(pipeline)
for doc in test:
    print(doc)"""

#close the mongodb connection
client.close()

#print the time used to make the aggregation
sec = end.tm_sec - st.tm_sec + 60*(end.tm_min - st.tm_min)
sec_res = sec%60
minu = (sec-sec_res)/60
print("temps de run : ",minu,"minutes et ",sec_res," secondes")
