# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:24:00 2024

@author: Nathan
"""
import pandas as pd
import json

data = []
with open('raw0.json', encoding="utf8") as f:
    x = f.readlines()
    
for i in range(len(x)):
    data.append(json.loads(x[i]))
    
data = pd.json_normalize(data)

"""Count NaN Value by columns"""
col = data.columns
Nacount = []
for i in col:
    Nacount.append(data[i].isna().sum())
    
Stud = pd.DataFrame(col,columns=["Attribut"])
"NA count "
NA_lim = 1600
Stud["NA_count"] = Nacount
Stud["NA_filter"] = Stud.NA_count.apply(lambda x :1  if x >NA_lim  else 0)
"""User attribute"""
Stud["user_filter"] = Stud.Attribut.apply(lambda x :1  if x.find("user.")==0  else 0)
user_attributes = Stud.query("user_filter==1").drop("NA_filter",axis=1)
"""Count point"""
Stud["lvl"] = Stud.Attribut.apply(lambda x :x.count("."))

Stud["origine"] = Stud.Attribut.apply(lambda x : x.split(".")[0]  if x.count(".")>0  else "/")


    
    

    
    



