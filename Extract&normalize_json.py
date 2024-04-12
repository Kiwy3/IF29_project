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
for i in col:
    c = data[i].isna().sum()
    
    

    
    



