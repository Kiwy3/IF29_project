"""

Neural network to classify user
In the case  study of IF29 class
author : Nathan Davouse

"""

import matplotlib.pyplot as plt
#connect from Mongo DB and import it on pandas
from pymongo import MongoClient
import pandas as pd
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_label #whole database
data = pd.DataFrame(list(collec.find()))

#Normalize data
features = ["verified", "friend_nb", "listed_nb", "follower_nb", 
            "favorites_nb", "len_description",
            "tweet_nb","hash_avg","at_avg","tweet_user_count",
            'tweet_frequency', 'friend_frequency',"visibility","Aggressivity"]
X = data[features]
Y = data[["label"]]

#normalize data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
X_sc = scaler.fit_transform(X)

#remove Vp & Ap from training
X_plot = X_sc[["visibility","Aggressivity"]].copy()
X_sc = X_sc.drop(["visibility","Aggressivity"],axis=1)
#Slice and correct label
X_train = X_sc[Y.label!=1]
Y_train = Y[Y.label !=1]
Y_train[Y_train.label == 2] = 1
#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.33)

"""Create and train the neural network"""

""" Import librairies """
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

"""Complex model
model = keras.Sequential([
    layers.BatchNormalization(input_shape=[len(features)]),
    #Next layer
    layers.Dense(units=256, activation='relu'),
    layers.BatchNormalization(),
    #Next layer
    layers.Dense(units=256, activation='relu'),
    layers.BatchNormalization(),
    #Output layer
    layers.Dense(1,activation= 'sigmoid')
])
"""

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[len(features)-2]),
    #Next layer
    layers.Dense(units=10, activation='relu'),
    layers.BatchNormalization(),
    #Output layer
    layers.Dense(1,activation= 'sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=256,
    epochs=50, 
    verbose = 1,
    callbacks=[early_stopping]
)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()


X_sc["predict"] = model.predict(X_sc)
def fun(x):
    if x>0.5 : return 1
    else : return 0
X_sc["new_label"] = X_sc["predict"].apply(fun)


plt.scatter(X_plot.visibility[X_sc["new_label"]==0],X_plot.Aggressivity[X_sc["new_label"]==0],s=0.5,c="blue",label = "non suspicious")
plt.scatter(X_plot.visibility[X_sc["new_label"]==1],X_plot.Aggressivity[X_sc["new_label"]==1],s=0.5,c="red",label = "suspicious")

plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.show()

