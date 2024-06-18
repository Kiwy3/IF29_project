"""

Neural network to classify user
In the case  study of IF29 class
author : Nathan Davouse

"""

"""---------------- Import librairies ----------------"""
#Plotting librairies
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Data management librairies
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
#Librairies for neural network
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

"""---------------- Import datas ----------------"""
#Principal databases
client = MongoClient("localhost", 27017)
db = client["IF29"]
collec = db.user_db_pca #whole database
data = pd.DataFrame(list(collec.find()))
#Obtain id and labels fields
labels = pd.DataFrame(list(db.user_label.find(projection = ["label"])))
id_list = data.pop("_id")
#manage X and Y table
X = data
X.set_index(id_list)
Y = labels["label"]
n_features = X.shape[1]

"""---------------- Slice and split databases ----------------"""
#Slice and correct label
X_train = X[Y!=0]
Y_train = Y[Y!=0]
Y_train[Y_train == -1] = 0
#Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.3,random_state=165464,shuffle=True)
X_train, X_val, Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.28,random_state=165464,shuffle=True)

"""----------------Create the IF29_01 model ----------------"""
def model_01():
    model = keras.Sequential([
        layers.Input(shape = (n_features,),name = "input"),
        layers.Dense(units=1, activation='relu',name="dense_layer"),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_classif_model_01"
    return model

"""----------------Create the IF29_02 model ----------------"""
def model_02():
    model = keras.Sequential([
        layers.Input(shape = (n_features,),name = "input"),
        layers.Dense(units=10, activation='relu',name="dense_layer_1"),
        layers.Dense(units=10, activation='relu',name="dense_layer_2"),
        #Output layer
        layers.Dense(1,activation= 'sigmoid',name = "output")
    ])
    model._name = "IF29_classif_model_02"
    return model
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


"""---------------- Choose and train the model ----------------"""
#Call model and print the summary
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
    verbose = 1,
    callbacks=[early_stopping]
)
#Save the model and history
model.save("./NN_models/"+model._name+".keras")
history_df = pd.DataFrame(history.history)

"""---------------- Look at the results of the model ----------------"""
#Make the prediction
X["Probability"] = model.predict(X)
def fun(x):
    if x>0.5 : return 1
    else : return 0
X["Prediction"] = X["Probability"].apply(fun)

#Indicators
X["label"] = Y
Test_subset = X.loc[X_test.index.tolist()]
Test_subset[Test_subset.label == -1] = 0
conf = confusion_matrix(Test_subset.label, Test_subset.Prediction)

#Save the output of the model
X_save = X
X_save.insert(0,"_id",id_list)
db.nn_output.drop()
db.nn_output.insert_many(X_save.to_dict('records'))

"""---------------- Plotting ----------------"""
#Plotting the history of the training
fig, ax = plt.subplots(2)
fig.suptitle("Evolution des indicateurs au cours de l'apprentissage")
ax[0].plot(history_df.loc[:, ['loss', 'val_loss']],label = ['loss', 'val_loss'])
ax[0].legend()
ax[0].set(ylabel = "Loss")
ax[1].plot(history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']],label =['binary_accuracy', 'val_binary_accuracy'] )
ax[1].legend()
ax[1].set(xlabel = "epochs",ylabel = "Entropy for binary accuracy")
fig.savefig("./images/5_3.Apprentissage_"+model._name+".png")
plt.show()

#Plot output of the network
fig, ax = plt.subplots(2)
fig.suptitle("Sortie du réseaux de neurones")
ax[0].scatter(X.pca0[X["Prediction"]==0],X.pca1[X["Prediction"]==0],s=0.5,c="blue",label = "non suspicious")
ax[0].scatter(X.pca0[X["Prediction"]==1],X.pca1[X["Prediction"]==1],s=0.5,c="red",label = "suspicious")
ax[0].legend()
ax[0].set(ylabel = "1ere composante")
predict_hist = ax[1].hist(X.Probability )
ax[1].vlines(0.5,0,max(predict_hist[0]),colors="red")
ax[1].set(xlabel = "probability of being suspicious",ylabel = "Number of observation")
fig.savefig("./images/5_3.network_output_"+model._name+".png")
plt.show()


#Plot the confusion matrix
fig, ax = plt.subplots()
display = ConfusionMatrixDisplay(conf)
ax.set(title = "Confusion matrix for NN model "+model._name)
display.plot(ax=ax)
fig.savefig("./images/5_3.confusion_matrix_"+model._name+".png")
plt.show()

#Plot results of the prediction
plt.scatter(X.pca0[X["Prediction"]==0],X.pca1[X["Prediction"]==0],s=0.5,c="blue",label = "non suspicious")
plt.scatter(X.pca0[X["Prediction"]==1],X.pca1[X["Prediction"]==1],s=0.5,c="red",label = "suspicious")
plt.legend()
plt.xlabel("1ere composante")
plt.ylabel("2eme composante")
plt.title("Répartition sur les premières composantes de l'ACP")
plt.savefig("./images/5_3.NN_results_"+model._name+".png")
plt.show()

