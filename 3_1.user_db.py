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
max_date = datetime(2018, 7, 18) #one day after last tweet

#caracter costs
hashtag_cost = 7.84
url_cost = 22.9
mention_cost = 10.57



#Make the aggregation pipeline
pipeline = [
    #{"$sort":{"current_time":1}}, #to allow $last to effectively be the last
    {"$limit": 50}, #if we need to test it on few lines
    {"$addFields":{"tweet_len_description" : { "$strLenCP": {"$ifNull" : ["$user.description",""] }}}},

    #Count specific caracter inside the text of a tweet, using entities attribute
    {"$addFields":{"tweet_hashtag_count" : {"$cond": { "if" : {"$isArray" : "$entities.hashtags"},"then" : {"$size":"$entities.hashtags"},"else" : 0}}},  },
    {"$addFields":{"tweet_mention_count" : {"$cond": { "if" : {"$isArray" : "$entities.user_mentions"},"then" : {"$size":"$entities.user_mentions"},"else" : 0}}},  },
    {"$addFields":{"tweet_urls_count" : {"$cond": { "if" : {"$isArray" : "$entities.urls"},"then" : {"$size":"$entities.urls"},"else" : 0}}},  },
    {"$addFields":{"tweet_symbols_count" : {"$cond": { "if" : {"$isArray" : "$entities.symbols"},"then" : {"$size":"$entities.symbols"},"else" : 0}}},  },
    #define the user date as a Date
    {"$addFields":{"user_date" : {"$dateFromString":{"dateString" : "$user.created_at"}}  }},
    #group by user 
    {"$group":{ #user global information
               "_id":"$user.id",
               "name" : {"$last":"$user.name"},
               "created_date" : {"$last":"$user_date"},
               #user boolean
               "verified":{"$max":"$user.verified"},
               #user numeric information
               "friend_nb":{"$avg":"$user.friends_count"}, #Follow someone
               "listed_nb":{"$avg":"$user.listed_count"},
               "follower_nb":{"$avg":"$user.followers_count"}, #is followed
               "favorites_nb":{"$avg":"$user.favourites_count"},
               "len_description" : {"$last" : "$tweet_len_description"},
                #Specific caracters count
               "hash_avg" : {"$avg" : "$tweet_hashtag_count"},
               "mention_avg" : {"$avg" : "$tweet_mention_count"},
               "url_avg" : {"$avg" : "$tweet_urls_count"},
               #tweet stats
               "tweet_nb" : {"$sum":1},
               "tweet_user_count":{"$max":"$user.statuses_count"} 
               }},
    #length of the user description
    #User lifetime used to make new attribute
    {"$addFields": { "user_lifetime":{"$dateDiff": {"startDate" : "$created_date",
                                   "endDate" : max_date,
                                   "unit" : "day"}}     }},
    # Add aggressivity
    {"$addFields" : {"tweet_frequency" : {"$divide" : ["$tweet_user_count","$user_lifetime"] }}},
    {"$addFields" : {"friend_frequency" : {"$divide" : ["$friend_nb","$user_lifetime"] }}},
    {"$addFields" : {"aggressivity" : {"$add" : ["$tweet_frequency","$friend_frequency"] }}},
    #Add visibility
    {"$addFields" : {"visi_1" : {"$multiply" : ["$mention_avg",mention_cost] }}},
    {"$addFields" : {"visi_2" : {"$multiply" : ["$hash_avg",hashtag_cost] }}},
    {"$addFields" : {"visi_3" : {"$multiply" : ["$url_avg",url_cost] }}},
    {"$addFields" : {"visibility" : {"$add" :["$visi_1","$visi_2","$visi_3"]}}},
    #Make the visibility attribute
    #{"$addFields" : {"visibility" : {"$add" :["$mention_avg"*mention_cost,"$hash_avg"*hashtag_cost,"$url_avg"*url_cost]}}},
    #create a ff_ratio attribute
    {"$addFields" : {"ff_ratio" : {"$cond" : { "if" : { "$eq": [ "$follower_nb", 0 ] },"then" : "$friend_nb","else" : {"$divide" : ["$friend_nb","$follower_nb"] }
    }  }}},
    {"$project":{"_id":1,"aggressivity":1,"visibility":1,"verified":1,"friend_nb":1,"listed_nb":1,"follower_nb":1,
                 "follower_nb":1,"favorites_nb":1,"len_description":1,"tweet_nb":1,"tweet_user_count":1,
                 "user_lifetime":1,"ff_ratio":1}},

    #{"$addFields" : {"ff_ratio" : {"$divide" : ["$friend_nb","$follower_nb"] }}},
    #Export it on another database
    #{"$out" : "user_db_sample"} #Sample database for small test
    {"$out" : "user_db"}
]

st = time.localtime() #to collect the time of start
collec.aggregate(pipeline)
end = time.localtime() #to collect the time of end of the aggregation

#close the mongodb connection
client.close()

#print the time used to make the aggregation
sec = end.tm_sec - st.tm_sec + 60*(end.tm_min - st.tm_min)
sec_res = sec%60
minu = (sec-sec_res)/60
print("temps de run : ",minu,"minutes et ",sec_res," secondes")
