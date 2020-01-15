# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:16:04 2019

@author: Tijev
"""

#%% data visualization
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
import pickle

#%% Importing the dataset
data = pd.read_csv("C:\\Users\\Tijev\\OneDrive\\Documenten\\School\\Thesis\\dataset.csv")
data['event_timestamp'] = pd.to_datetime(data.event_timestamp)
data = data.sort_values('event_timestamp')

data.groupby('message_severity').count()
data.groupby('an_description').count()
data.groupby('an_title').count()
data.groupby('an_robot').count()
data.groupby('an_line').count()
data.groupby('an_cell').count()

#%% Cleaning the NaN values
"""NaN values will be replaced by the most common value in the column
"""

data["message_category"].value_counts()
data = data.fillna({"message_category": "Motion"})

data["an_title"].value_counts()
data = data.fillna({"an_title": "Corner path failure"})

data["an_description"].value_counts()
data = data.fillna({"an_description": "The regain movement has started"})



#%% Transforming categorical values into numerical values


data["message_severity"] = data["message_severity"].astype('category')
data.dtypes
data["message_severity"] = data["message_severity"].cat.codes

data["an_line"] = data["an_line"].astype('category')
data.dtypes
data["an_line"] = data["an_line"].cat.codes

data["an_cell"] = data["an_cell"].astype('category')
data.dtypes
data["an_cell"] = data["an_cell"].cat.codes

data["an_robot"] = data["an_robot"].astype('category')
data.dtypes
data["an_robot"] = data["an_robot"].cat.codes

#%%

data['event_timestamp'] = pd.to_datetime(data.event_timestamp)
data = data.sort_values('event_timestamp')


data = data.drop(['Unnamed: 0', 'message_category', 'an_title', 'an_description'], axis=1)


#%% This script will sample 

pickle_in = open("dataarray.pickle","rb")
dataarray = pickle.load(pickle_in) 
    
listwitheventswitherror_timed = []

for index, array in enumerate(dataarray):
    if array[1] == 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=27)
        sample = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample = sample.drop(['message_severity', 'event_timestamp'], axis=1)
        if len(sample) < 100 and len(sample) > 1:
            listwitheventswitherror_timed.append(sample)
            print(sample)
      
listwitheventswithnoerror_timed = []        
        
for index, array in enumerate(dataarray[5000:]):
    if array[1] != 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=27)
        sample2 = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample2 = sample2.drop(['message_severity', 'event_timestamp'], axis=1)
        
        if len(sample2) < 100 and len(sample2) > 1:
            listwitheventswithnoerror_timed.append(sample2)
            print(sample2)
            
    if len(listwitheventswithnoerror_timed) == len(listwitheventswitherror_timed):
        break

templist = []
for sequence in listwitheventswitherror_timed:
    sequence = np.asarray(sequence)
    templist.append(sequence)
  
templist2 = []
for sequence in listwitheventswithnoerror_timed:
    sequence = np.asarray(sequence)
    templist2.append(sequence)    
    

    


         
#%% Deleting empty arrays and adding labels
import random
emptylist1 = []
for index, row in enumerate(listwitheventswitherror_timed):
    values = listwitheventswitherror_timed[index].values
    emptylist1.append(values)    
      
    
emptylist2 = []
for index, row in enumerate(listwitheventswithnoerror_timed):
    values = listwitheventswithnoerror_timed[index].values
    emptylist2.append(values)


    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror_timed):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror_timed):
    added = np.append(array, tagnoerror)
    addedarraywithnoerror.append(added)
addedarraywithnoerror = np.asarray(addedarraywithnoerror)

len(max(addedarraywitherror, key=len))
balancedlist = []

for item in addedarraywitherror:
    balancedlist.append(item)
    
for item in addedarraywithnoerror:
    balancedlist.append(item)
    
    
for item in balancedlist:
    print(item[-1])
    
balancedlist[100]


    
#%%splitting into X and Y
X = []
for row in balancedlist:
    array = row[:-1]
    X.append(array)
    
Y = []
for row in balancedlist:
    array = row[-1]
    Y.append(array)


X = np.asarray(X)
Y = np.asarray(Y)


#%% Zero-padding the results and normalizing
    
biggestarray = max(balancedlist, key=len)
len(biggestarray)

listpad = []
listpad2 = []


def padding(array, length):
    list1 = []
    for id, value in enumerate(X):
            npad = length - len(value)
            output = np.pad(value, pad_width=((0,npad)), mode='constant')
            list1.append(output)
    return np.asarray(list1)

Xpadded = padding(X, len(biggestarray))

biggestarray = max(Xpadded, key=len) #checking the lengths
len(biggestarray)
       
              


Xreshaped = Xpadded.reshape((len(Xpadded),len(biggestarray),1)) #match the shape of your sample size

standardizedlist = []
scaler = StandardScaler()
for timeframe in Xreshaped:
    scaler.fit(timeframe)
    scaled = scaler.transform(timeframe)
    standardizedlist.append(scaled)
    
Xreshaped = np.asarray(standardizedlist)
#%% Splitting into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
     Xreshaped, Y, test_size=0.20, random_state=42)



#%% Building a Recursive Neural Network

from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.models import Sequential
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import keras
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#%% Testing Cuda
class TestCudnnLSTM():

  def __init__(self):  
    self.max_length = 1000
    self.n_input_dim = 1    

    self.model = []
    
    self.config()
    self.create_model()
    
  def config(self):
    print("Keras version: " + keras.__version__)
    print("Tensorflow version: " + tf.__version__)
    
    config = tf.ConfigProto()
    return config
    
  def create_model(self):        
          
    print('Creating Model')
    model = Sequential()
    model.add(CuDNNLSTM(1,
                    return_sequences=True,
                    stateful=False,
                    kernel_initializer='he_normal',
                    input_shape=(self.max_length, self.n_input_dim)))
    print (model.summary())
    
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  weighted_metrics=['accuracy'],
                  sample_weight_mode='temporal')
  
    print('Model compiled')      
    self.model = model
    return self
      
    
if __name__ == "__main__":
  mt = TestCudnnLSTM()
  
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
TestCudnnLSTM()
#%%
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(len(biggestarray),1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0001, decay=0.000001)

"""
use this part below to reload a model thats been saved by checkpoint
"""
#load weights (optional)
model.load_weights("best_model_timed1.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_timed1.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_timed1.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 128, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1)

#%% evaluating performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
        
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)

conf_matrix = confusion_matrix(y_test, pred_list, labels=None, sample_weight=None)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

report = classification_report(y_test, pred_list)
print(report)

#%%

listwitheventswitherror_timed = []

for index, array in enumerate(dataarray):
    if array[1] == 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=36)
        sample = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample = sample.drop(['message_severity', 'event_timestamp'], axis=1)
        if len(sample) < 100 and len(sample) > 1:
            listwitheventswitherror_timed.append(sample)
            print(sample)
      
listwitheventswithnoerror_timed = []        
        
for index, array in enumerate(dataarray[5000:]):
    if array[1] != 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=36)
        sample2 = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample2 = sample2.drop(['message_severity', 'event_timestamp'], axis=1)
        
        if len(sample2) < 100 and len(sample2) > 1:
            listwitheventswithnoerror_timed.append(sample2)
            print(sample2)
            
    if len(listwitheventswithnoerror_timed) == len(listwitheventswitherror_timed):
        break



templist = []
for sequence in listwitheventswitherror_timed:
    sequence = np.asarray(sequence)
    templist.append(sequence)
  
templist2 = []
for sequence in listwitheventswithnoerror_timed:
    sequence = np.asarray(sequence)
    templist2.append(sequence)    
    


         
#%% Deleting empty arrays and adding labels
import random
emptylist1 = []
for index, row in enumerate(listwitheventswitherror_timed):
    values = listwitheventswitherror_timed[index].values
    emptylist1.append(values)    
      
    
emptylist2 = []
for index, row in enumerate(listwitheventswithnoerror_timed):
    values = listwitheventswithnoerror_timed[index].values
    emptylist2.append(values)


    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror_timed):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror_timed):
    added = np.append(array, tagnoerror)
    addedarraywithnoerror.append(added)
addedarraywithnoerror = np.asarray(addedarraywithnoerror)

len(max(addedarraywitherror, key=len))
balancedlist = []

for item in addedarraywitherror:
    balancedlist.append(item)
    
for item in addedarraywithnoerror:
    balancedlist.append(item)
    
    
for item in balancedlist:
    print(item[-1])
    
balancedlist[2]


    
#%%splitting into X and Y
X = []
for row in balancedlist:
    array = row[:-1]
    X.append(array)
    
Y = []
for row in balancedlist:
    array = row[-1]
    Y.append(array)


X = np.asarray(X)
Y = np.asarray(Y)


#%% Zero-padding the results and normalizing
    
biggestarray = max(balancedlist, key=len)
len(biggestarray)

listpad = []
listpad2 = []


def padding(array, length):
    list1 = []
    for id, value in enumerate(X):
            npad = length - len(value)
            output = np.pad(value, pad_width=((0,npad)), mode='constant')
            list1.append(output)
    return np.asarray(list1)

Xpadded = padding(X, len(biggestarray))

biggestarray = max(Xpadded, key=len) #checking the lengths
len(biggestarray)
       
              


Xreshaped = Xpadded.reshape((len(Xpadded),len(biggestarray),1)) #match the shape of your sample size

standardizedlist = []
scaler = StandardScaler()
for timeframe in Xreshaped:
    scaler.fit(timeframe)
    scaled = scaler.transform(timeframe)
    standardizedlist.append(scaled)
    
Xreshaped = np.asarray(standardizedlist)
#%% Splitting into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
     Xreshaped, Y, test_size=0.20, random_state=42)


#%%
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(len(biggestarray),1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0001, decay=0.000001)

"""
use this part below to reload a model thats been saved by checkpoint
"""
#load weights (optional)
model.load_weights("best_model_timed2.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_timed2.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_timed2.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 128, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1)


#%% evaluating performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
        
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)

conf_matrix = confusion_matrix(y_test, pred_list, labels=None, sample_weight=None)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

report = classification_report(y_test, pred_list)
print(report)


#%%
listwitheventswitherror_timed = []

for index, array in enumerate(dataarray):
    if array[1] == 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=43)
        sample = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample = sample.drop(['message_severity', 'event_timestamp'], axis=1)
        if len(sample) < 100 and len(sample) > 1:
            listwitheventswitherror_timed.append(sample)
            print(sample)
      
listwitheventswithnoerror_timed = []        
        
for index, array in enumerate(dataarray[5000:]):
    if array[1] != 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=43)
        sample2 = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample2 = sample2.drop(['message_severity', 'event_timestamp'], axis=1)
        
        if len(sample2) < 100 and len(sample2) > 1:
            listwitheventswithnoerror_timed.append(sample2)
            print(sample2)
            
    if len(listwitheventswithnoerror_timed) == len(listwitheventswitherror_timed):
        break
    
templist = []
for sequence in listwitheventswitherror_timed:
    sequence = np.asarray(sequence)
    templist.append(sequence)
  
templist2 = []
for sequence in listwitheventswithnoerror_timed:
    sequence = np.asarray(sequence)
    templist2.append(sequence)    
    


         
#%% Deleting empty arrays and adding labels
import random
emptylist1 = []
for index, row in enumerate(listwitheventswitherror_timed):
    values = listwitheventswitherror_timed[index].values
    emptylist1.append(values)    
      
    
emptylist2 = []
for index, row in enumerate(listwitheventswithnoerror_timed):
    values = listwitheventswithnoerror_timed[index].values
    emptylist2.append(values)


    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror_timed):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror_timed):
    added = np.append(array, tagnoerror)
    addedarraywithnoerror.append(added)
addedarraywithnoerror = np.asarray(addedarraywithnoerror)

len(max(addedarraywitherror, key=len))
balancedlist = []

for item in addedarraywitherror:
    balancedlist.append(item)
    
for item in addedarraywithnoerror:
    balancedlist.append(item)
    

    
for item in balancedlist:
    print(item[-1])
    
balancedlist[2]


    
#%%splitting into X and Y
X = []
for row in balancedlist:
    array = row[:-1]
    X.append(array)
    
Y = []
for row in balancedlist:
    array = row[-1]
    Y.append(array)


X = np.asarray(X)
Y = np.asarray(Y)


#%% Zero-padding the results and normalizing
    
biggestarray = max(balancedlist, key=len)
len(biggestarray)

listpad = []
listpad2 = []


def padding(array, length):
    list1 = []
    for id, value in enumerate(X):
            npad = length - len(value)
            output = np.pad(value, pad_width=((0,npad)), mode='constant')
            list1.append(output)
    return np.asarray(list1)

Xpadded = padding(X, len(biggestarray))

biggestarray = max(Xpadded, key=len) #checking the lengths
len(biggestarray)
       
              


Xreshaped = Xpadded.reshape((len(Xpadded),len(biggestarray),1)) #match the shape of your sample size

standardizedlist = []
scaler = StandardScaler()
for timeframe in Xreshaped:
    scaler.fit(timeframe)
    scaled = scaler.transform(timeframe)
    standardizedlist.append(scaled)
    
Xreshaped = np.asarray(standardizedlist)
#%% Splitting into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
     Xreshaped, Y, test_size=0.20, random_state=42)

#%%
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(len(biggestarray),1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0001, decay=0.000001)

"""
use this part below to reload a model thats been saved by checkpoint
"""
#load weights (optional)
#model.load_weights("best_model_timed3.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_timed3.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_timed3.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 128, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1)

#%% evaluating performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
        
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)

conf_matrix = confusion_matrix(y_test, pred_list, labels=None, sample_weight=None)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

report = classification_report(y_test, pred_list)
print(report)

#%%
listwitheventswitherror_timed = []

for index, array in enumerate(dataarray):
    if array[1] == 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=50)
        sample = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample = sample.drop(['message_severity', 'event_timestamp'], axis=1)
        if len(sample) < 100 and len(sample) > 1:
            listwitheventswitherror_timed.append(sample)
            print(sample)
      
listwitheventswithnoerror_timed = []        
        
for index, array in enumerate(dataarray[5000:]):
    if array[1] != 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=50)
        sample2 = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample2 = sample2.drop(['message_severity', 'event_timestamp'], axis=1)
        
        if len(sample2) < 100 and len(sample2) > 1:
            listwitheventswithnoerror_timed.append(sample2)
            print(sample2)
            
    if len(listwitheventswithnoerror_timed) == len(listwitheventswitherror_timed):
        break

templist = []
for sequence in listwitheventswitherror_timed:
    sequence = np.asarray(sequence)
    templist.append(sequence)
  
templist2 = []
for sequence in listwitheventswithnoerror_timed:
    sequence = np.asarray(sequence)
    templist2.append(sequence)    
    


         
#%% Deleting empty arrays and adding labels
import random
emptylist1 = []
for index, row in enumerate(listwitheventswitherror_timed):
    values = listwitheventswitherror_timed[index].values
    emptylist1.append(values)    
      
    
emptylist2 = []
for index, row in enumerate(listwitheventswithnoerror_timed):
    values = listwitheventswithnoerror_timed[index].values
    emptylist2.append(values)


    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror_timed):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror_timed):
    added = np.append(array, tagnoerror)
    addedarraywithnoerror.append(added)
addedarraywithnoerror = np.asarray(addedarraywithnoerror)

len(max(addedarraywitherror, key=len))
balancedlist = []

for item in addedarraywitherror:
    balancedlist.append(item)
    
for item in addedarraywithnoerror:
    balancedlist.append(item)

    
for item in balancedlist:
    print(item[-1])
    
balancedlist[2]


    
#%%splitting into X and Y
X = []
for row in balancedlist:
    array = row[:-1]
    X.append(array)
    
Y = []
for row in balancedlist:
    array = row[-1]
    Y.append(array)


X = np.asarray(X)
Y = np.asarray(Y)


#%% Zero-padding the results and normalizing
    
biggestarray = max(balancedlist, key=len)
len(biggestarray)

listpad = []
listpad2 = []


def padding(array, length):
    list1 = []
    for id, value in enumerate(X):
            npad = length - len(value)
            output = np.pad(value, pad_width=((0,npad)), mode='constant')
            list1.append(output)
    return np.asarray(list1)

Xpadded = padding(X, len(biggestarray))

biggestarray = max(Xpadded, key=len) #checking the lengths
len(biggestarray)
       
              


Xreshaped = Xpadded.reshape((len(Xpadded),len(biggestarray),1)) #match the shape of your sample size

standardizedlist = []
scaler = StandardScaler()
for timeframe in Xreshaped:
    scaler.fit(timeframe)
    scaled = scaler.transform(timeframe)
    standardizedlist.append(scaled)
    
Xreshaped = np.asarray(standardizedlist)
#%% Splitting into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
     Xreshaped, Y, test_size=0.20, random_state=42)

#%%
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(len(biggestarray),1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0001, decay=0.000001)

"""
use this part below to reload a model thats been saved by checkpoint
"""
#load weights (optional)
model.load_weights("best_model_timed4.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_timed4.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_timed4.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 64, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1)


#%% evaluating performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
        
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)

conf_matrix = confusion_matrix(y_test, pred_list, labels=None, sample_weight=None)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

report = classification_report(y_test, pred_list)
print(report)

#%%

listwitheventswitherror_timed = []

for index, array in enumerate(dataarray):
    if array[1] == 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=58)
        sample = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample = sample.drop(['message_severity', 'event_timestamp'], axis=1)
        if len(sample) < 100 and len(sample) > 1:
            listwitheventswitherror_timed.append(sample)
            print(sample)
      
listwitheventswithnoerror_timed = []        
        
for index, array in enumerate(dataarray[5000:]):
    if array[1] != 71058:
        event_timestamp = array[0]
        prediction_window = event_timestamp - pd.Timedelta(minutes=58)
        sample2 = data.loc[(data['event_timestamp'] > (prediction_window - pd.Timedelta(minutes=30))) & (data['event_timestamp'] < prediction_window)]
        sample2 = sample2.drop(['message_severity', 'event_timestamp'], axis=1)
        
        if len(sample2) < 100 and len(sample2) > 1:
            listwitheventswithnoerror_timed.append(sample2)
            print(sample2)
            
    if len(listwitheventswithnoerror_timed) == len(listwitheventswitherror_timed):
        break


templist = []
for sequence in listwitheventswitherror_timed:
    sequence = np.asarray(sequence)
    templist.append(sequence)
  
templist2 = []
for sequence in listwitheventswithnoerror_timed:
    sequence = np.asarray(sequence)
    templist2.append(sequence)    
    


         
#%% Deleting empty arrays and adding labels
import random
emptylist1 = []
for index, row in enumerate(listwitheventswitherror_timed):
    values = listwitheventswitherror_timed[index].values
    emptylist1.append(values)    
      
    
emptylist2 = []
for index, row in enumerate(listwitheventswithnoerror_timed):
    values = listwitheventswithnoerror_timed[index].values
    emptylist2.append(values)


    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror_timed):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror_timed):
    added = np.append(array, tagnoerror)
    addedarraywithnoerror.append(added)
addedarraywithnoerror = np.asarray(addedarraywithnoerror)

len(max(addedarraywitherror, key=len))
balancedlist = []

for item in addedarraywitherror:
    balancedlist.append(item)
    
for item in addedarraywithnoerror:
    balancedlist.append(item)

    
for item in balancedlist:
    print(item[-1])
    
balancedlist[2]


    
#%%splitting into X and Y
X = []
for row in balancedlist:
    array = row[:-1]
    X.append(array)
    
Y = []
for row in balancedlist:
    array = row[-1]
    Y.append(array)


X = np.asarray(X)
Y = np.asarray(Y)


#%% Zero-padding the results and normalizing
    
biggestarray = max(balancedlist, key=len)
len(biggestarray)

listpad = []
listpad2 = []


def padding(array, length):
    list1 = []
    for id, value in enumerate(X):
            npad = length - len(value)
            output = np.pad(value, pad_width=((0,npad)), mode='constant')
            list1.append(output)
    return np.asarray(list1)

Xpadded = padding(X, len(biggestarray))

biggestarray = max(Xpadded, key=len) #checking the lengths
len(biggestarray)
       
              


Xreshaped = Xpadded.reshape((len(Xpadded),len(biggestarray),1)) #match the shape of your sample size

standardizedlist = []
scaler = StandardScaler()
for timeframe in Xreshaped:
    scaler.fit(timeframe)
    scaled = scaler.transform(timeframe)
    standardizedlist.append(scaled)
    
Xreshaped = np.asarray(standardizedlist)
#%% Splitting into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
     Xreshaped, Y, test_size=0.20, random_state=42)


#%%
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(len(biggestarray),1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0001, decay=0.000001)

"""
use this part below to reload a model thats been saved by checkpoint
"""
#load weights (optional)
model.load_weights("best_model_timed5.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_timed5.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_timed5.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 128, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test, verbose=1)


#%% evaluating performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

pred_list = []

#turning predictions into results
for row in predictions:
    if row < 0.5:
        pred_list.append(0)
    elif row > 0.5:
        pred_list.append(1)
        
bool_list = list(map(bool,pred_list))
results = np.asarray(bool_list)

conf_matrix = confusion_matrix(y_test, pred_list, labels=None, sample_weight=None)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');

report = classification_report(y_test, pred_list)
print(report)

