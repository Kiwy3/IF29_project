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
features = ['verified', 'protected', 'friend_nb',
        'listed_nb', 'follower_nb', 'favorites_nb', 'len_description',
        'hash_avg', 'mention_avg', 'url_avg', 'symbols_avg', 'tweet_nb',
        'tweet_user_count', 'user_lifetime', 'tweet_frequency',
        'friend_frequency', 'aggressivity', 'visibility', 'ff_ratio']
X = data[features]
Y = data[["label"]]

#Slice and correct label
X_train = X[Y.label!=0]
X_pred = X[Y.label == 0]
Y_train = Y[Y.label !=0]
Y_train[Y_train.label == -1] = 0
#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.3,random_state=165464,shuffle=True)
X_train, X_val, Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.28,random_state=165464,shuffle=True)

""" Import librairies """
from tensorflow import keras
from tensorflow.keras import layers # type: ignore


"""----------------Create the IF29_01 model ----------------"""
def model_01():
    model = keras.Sequential([
        layers.Input(shape = (11,),name = "input"),
        layers.Dense(units=1, activation='relu',name="dense_layer"),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_01"
    return model

"""----------------Create the IF29_02 model ----------------"""
def model_02():
    model = keras.Sequential([
        layers.Input(shape = (len(features),),name = "input"),
        layers.Dense(units=10, activation='relu',name="dense_layer_1"),
        layers.Dense(units=10, activation='relu',name="dense_layer_2"),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_02"
    return model

"""----------------Create the IF29_03 model ----------------"""
def model_03():
    model = keras.Sequential([
        layers.Input(shape = (11,),name = "input"),
        layers.BatchNormalization(),
        layers.Dense(units=10, activation='relu',name="dense_layer_1"),
        layers.Dropout(rate=0.3),
        layers.BatchNormalization(),
        layers.Dense(units=10, activation='relu',name="dense_layer_2"),
        layers.BatchNormalization(),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_03"
    return model

def model_031():
    model = keras.Sequential([
        layers.Input(shape = (11,),name = "input"),
        layers.Dropout(rate=0.3),
        layers.Dense(units=10, activation='relu',name="dense_layer_1"),
        layers.Dropout(rate=0.3),
        layers.Dense(units=10, activation='relu',name="dense_layer_2"),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_031"
    return model

#choose model and show it
model = model_02()
model.summary()

#Compile to define the training of the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

#Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)

#Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=256,
    epochs=300, 
    verbose = 1
    ,callbacks=[early_stopping]
)

"""Plot evolution of the training"""
history_df = pd.DataFrame(history.history)
def history_plot():
    fig, ax = plt.subplots(2)
    fig.suptitle("Evolution des indicateurs au cours de l'apprentissage")
    ax[0].plot(history_df.loc[:, ['loss', 'val_loss']],label = ['loss', 'val_loss'])
    ax[0].legend()
    ax[0].set(ylabel = "Loss")
    ax[1].plot(history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']],label =['binary_accuracy', 'val_binary_accuracy'] )
    ax[1].legend()
    ax[1].set(xlabel = "epochs",ylabel = "Entropy for binary accuracy")
    plt.show()
history_plot()


X["predict"] = model.predict(X)
def fun(x):
    if x>0.5 : return 1
    else : return 0
X_pred["new_label"] = X_pred["predict"].apply(fun)


plt.scatter(X_pred.visibility[X_pred["new_label"]==0],X_pred.aggressivity[X_pred["new_label"]==0],s=0.5,c="blue",label = "non suspicious")
plt.scatter(X_pred.visibility[X_pred["new_label"]==1],X_pred.aggressivity[X_pred["new_label"]==1],s=0.5,c="red",label = "suspicious")

plt.legend()
plt.xlabel("visibility")
plt.ylabel("aggressivity")
plt.show()
