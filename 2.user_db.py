"""

Make user database from tweet database with mongo aggregate
In the case  study of IF29 class
author : Nathan Davouse

"""
#Import librairies
from pymongo import MongoClient
import time
from datetime import datetime

#Connection with mongoDB
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.tweets_global

#define the comparaison date
max_date = datetime(2018, 7, 1)

#Make the aggregation pipeline
pipeline = [
    {"$sort":{"current_time":1}}, #to allow $last to effectively be the last
    #{"$limit": 100}, #if we need to test it on few lines
     
    #Add nb of # and @ on each tweet
    {"$addFields":{"hash_count" : {"$subtract": [{"$size" : {"$split" : ["$text","#"]} },1 ]  }},  },
    {"$addFields":{"at_count" : {"$subtract": [{"$size" : {"$split" : ["$text","@"]} },1 ]  }},  },
    #manage text field
    {"$addFields":{"tweet_url_bool" : { "$cond" : [{"$eq" : ["$user.url","null"]},True,False]}},  },
    #{"$addFields":{"tweet_len_description" : { "$cond" : [{"$eq" : ["$user.description","null"]},{"$strLenBytes" : "$user.description"},0 ]}},  },
    #mange user.created_at date fields
    {"$addFields":{"user_date" : {"$dateFromString":{"dateString" : "$user.created_at"}}  }},
    #group by user 
    {"$group":{ #user global information
               "_id":"$user.id",
               "name" : {"$last":"$user.name"},
               "created_date" : {"$last":"$user_date"},
               "verified":{"$last":"$user.verified"},
               #user numeric information
               "friend_nb":{"$avg":"$user.friends_count"},
               "listed_nb":{"$avg":"$user.listed_count"},
               "follower_nb":{"$avg":"$user.followers_count"},
               "favorites_nb":{"$avg":"$user.favourites_count"},
               #Text Fields
               "url_bool" : {"$max" :"$tweet_url_bool" },
               "description" : {"$last" : "$user.description"},
               #tweet stats
               "tweet_nb" : {"$sum":1},
               "hash_avg" : {"$avg" : "$hash_count"},
               "at_avg" : {"$avg" : "$at_count"},
               "retweet_avg" : {"$avg" : "$retweet.count"},
               "tweet_user_count":{"$max":"$user.statuses_count"} 
               }},
    #Make the visibility attribute
    {"$addFields": { "user_lifetime":{"$dateDiff": {"startDate" : "$created_date",
                                   "endDate" : max_date,
                                   "unit" : "day"}}     }},
    {"$addFields" : {"tweet_frequency" : {"$divide" : ["$tweet_user_count","$user_lifetime"] }}},
    {"$addFields" : {"friend_frequency" : {"$divide" : ["$friend_nb","$user_lifetime"] }}},
    {"$addFields" : {"visibility" : {"$add" : ["$tweet_frequency","$friend_frequency"] }}},
    #Make the aggressivity attribute
    {"$addFields" : {"Aggressivity" : {"$add" : ["$at_avg","$hash_avg"] }}},
    #Export it on another database
     #,{"$out" : "user_db_sample"} #Sample database for small test
    {"$out" : "user_db_V2"}
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
