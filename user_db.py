from pymongo import MongoClient
import time

client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.wc_import
st = time.localtime()

pipeline = [
    {"$sort":{"current_time":1}},
    #{"$limit": 3},
    {"$addFields":{"hash_count" : {
                        "$subtract": [
                            {"$size" : {"$split" : ["$text","#"]} },1 ]  }
                   
                   }},
    {"$group":{"_id":"$user.id",
               "name" : {"$last":"$user.name"},
               "hash_nb" : {"$sum" : "$hash_count"},
               "hash_avg" : {"$avg" : "$hash_count"},
               "created_date" : {"$last":"$user.created_at"},
               "friend_nb":{"$avg":"$user.friends_count"},
               "listed_nb":{"$avg":"$user.listed_count"},
               "verified":{"$last":"$user.verified"},
               "follower_nb":{"$avg":"$user.followers_count"},
               "favorites_nb":{"$avg":"$user.favourites_count"},
               #"len_description" : {"$avg" : {"$strLenCP" : "$user.description"}},
               "tweet_nb":{"$max":"$user.statuses_count"},
               "url" : {"$max" :"$user.url" },
               "tweet_count" : {"$sum":1} 
               }}
    ,{"$out" : "export_test"}
    #,{"$limit":10}
]

collec.aggregate(pipeline)
#test = collec.aggregate(pipeline)
"""for doc in test:
    print(doc)"""
end = time.localtime()
 
sec = end.tm_sec - st.tm_sec + 60*(end.tm_min - st.tm_min)
sec_res = sec%60
minu = (sec-sec_res)/60

print("temps de run : ",minu,"minutes et ",sec_res," secondes")
