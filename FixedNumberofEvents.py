# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:16:04 2019

@author: Tijev
This script consists out of 5 parts which repeat itself. This script takes a fixed number of events. 
The script called "TimedEvents" takes a different approach and samples according to time.
Each part has its own pre-processing in order to sample and label the data.
This script runs 5 experiments in total. Each experiment has it's best model saved as a h5 file.
The file can be loaded into the model so that it doesn't have to be trained again.

"""

#%% importing packages for data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
import pickle





pickle_in = open("dataarray.pickle","rb")
dataarray = pickle.load(pickle_in) 
    


#%% Experiment [1/5]

listwitheventswithnoerror = []
listwitheventswitherror = []
lengthsequence = 60
"""-----------------------------------------------------------------"""


betweentimes = []

for index, array in enumerate(dataarray):
   if array[1] == 71058:
       randomnumber = random.randint(10, 30) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]   
       start_time = sample[-1][0]
       end_time = array[0]
       between_time = end_time - start_time
       betweentimes.append(between_time)
       if len(sample) > 1:
           listwitheventswitherror.append(sample)
           print(index)

for index, array in enumerate(dataarray):
   if array[1] != 71058:
       randomnumber = random.randint(10, 30) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]
       errorcounter = 0
       for sequence in dataarray[index:index+40]:
           if sequence[1] == 71058:
               errorcounter += 1
        
       if errorcounter == 0 and len(sample) > 1:
           listwitheventswithnoerror.append(sample)
           start_time = sample[0][0]
           end_time = array[0]
           between_time = end_time - start_time
           betweentimes.append(between_time)
           print(index)
   if len(listwitheventswithnoerror) > len(listwitheventswitherror):
       break

sample[-1]   
    

starttime = betweentimes[0]
for time in betweentimes[1:]:
    starttime = starttime + time
starttime / 2233


testlist = []
for index, sequence in enumerate(listwitheventswitherror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist.append(templist)
    
testlist2 = []
for index, sequence in enumerate(listwitheventswithnoerror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist2.append(templist)
        
listwitheventswitherror = testlist  
listwitheventswithnoerror = testlist2  
listwitheventswithnoerror[0] 
                    
#%% Deleting empty arrays and adding labels
import random
    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror):
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
model.load_weights("best_model_randomN2.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_randomN1.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_randomN1.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #loading the bestmodel checkpoint
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.1, shuffle=True, batch_size = 32, callbacks=callbacks_list)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy_window1.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_window1.png')
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
listwitheventswithnoerror = []
listwitheventswitherror = []
lengthsequence = 60
"""-----------------------------------------------------------------"""

for index, array in enumerate(dataarray):
   if array[1] == 71058:
       randomnumber = random.randint(30,50) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]             
       if len(sample) > 1:
           listwitheventswitherror.append(sample)
           print(index)

for index, array in enumerate(dataarray):
   if array[1] != 71058:
       randomnumber = random.randint(30,50) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]
       errorcounter = 0
       for sequence in dataarray[index:index+40]:
           if sequence[1] == 71058:
               errorcounter += 1
        
       if errorcounter == 0 and len(sample) > 1:
           listwitheventswithnoerror.append(sample)
           print(index)
   if len(listwitheventswithnoerror) > len(listwitheventswitherror):
       break

testlist = []
for index, sequence in enumerate(listwitheventswitherror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist.append(templist)
  

testlist2 = []
for index, sequence in enumerate(listwitheventswithnoerror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist2.append(templist)
        

listwitheventswithnoerror = testlist2
listwitheventswitherror = testlist

listwitheventswithnoerror[0]
listwitheventswitherror[0]



         
#%% Deleting empty arrays and adding labels
import random
    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror):
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
#model.load_weights("best_model_randomN.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_randomN2.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_randomN2.h5'
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
plt.savefig('accuracy_window2.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_window2.png')
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
listwitheventswithnoerror = []
listwitheventswitherror = []
lengthsequence = 60
"""-----------------------------------------------------------------"""

for index, array in enumerate(dataarray):
   if array[1] == 71058:
       randomnumber = random.randint(50,70) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]             
       if len(sample) > 1:
           listwitheventswitherror.append(sample)
           print(index)

for index, array in enumerate(dataarray):
   if array[1] != 71058:
       randomnumber = random.randint(50,70) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]
       errorcounter = 0
       for sequence in dataarray[index:index+40]:
           if sequence[1] == 71058:
               errorcounter += 1
        
       if errorcounter == 0 and len(sample) > 1:
           listwitheventswithnoerror.append(sample)
           print(index)
   if len(listwitheventswithnoerror) > len(listwitheventswitherror):
       break

testlist = []
for index, sequence in enumerate(listwitheventswitherror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist.append(templist)
    
testlist2 = []
for index, sequence in enumerate(listwitheventswithnoerror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist2.append(templist)
        

listwitheventswitherror = testlist
listwitheventswithnoerror = testlist2

listwitheventswitherror[0]
listwitheventswithnoerror[0]


         
#%% Deleting empty arrays and adding labels
import random
    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror):
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
#model.load_weights("best_model_randomN.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_randomN3.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_randomN3.h5'
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
plt.savefig('accuracy_window3.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_window3.png')
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

listwitheventswithnoerror = []
listwitheventswitherror = []
lengthsequence = 60
"""-----------------------------------------------------------------"""

for index, array in enumerate(dataarray):
   if array[1] == 71058:
       randomnumber = random.randint(70,90) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]             
       if len(sample) > 1:
           listwitheventswitherror.append(sample)
           print(index)

for index, array in enumerate(dataarray):
   if array[1] != 71058:
       randomnumber = random.randint(70,90) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]
       errorcounter = 0
       for sequence in dataarray[index:index+40]:
           if sequence[1] == 71058:
               errorcounter += 1
        
       if errorcounter == 0 and len(sample) > 1:
           listwitheventswithnoerror.append(sample)
           print(index)
   if len(listwitheventswithnoerror) > len(listwitheventswitherror):
       break

testlist = []
for index, sequence in enumerate(listwitheventswitherror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist.append(templist)
    
testlist2 = []
for index, sequence in enumerate(listwitheventswithnoerror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist2.append(templist)


listwitheventswitherror = testlist
listwitheventswithnoerror = testlist2

         
#%% Deleting empty arrays and adding labels
import random
    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror):
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
#model.load_weights("best_model_randomN.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_randomN4.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_randomN4.h5'
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
plt.savefig('accuracy_window4.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_window4.png')
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

listwitheventswithnoerror = []
listwitheventswitherror = []
lengthsequence = 60
"""-----------------------------------------------------------------"""

for index, array in enumerate(dataarray):
   if array[1] == 71058:
       randomnumber = random.randint(90,110) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]             
       if len(sample) > 1:
           listwitheventswitherror.append(sample)
           print(index)

for index, array in enumerate(dataarray):
   if array[1] != 71058:
       randomnumber = random.randint(90,110) 
       sample = dataarray[index-lengthsequence-randomnumber:index-10-randomnumber]
       errorcounter = 0
       for sequence in dataarray[index:index+40]:
           if sequence[1] == 71058:
               errorcounter += 1
        
       if errorcounter == 0 and len(sample) > 1:
           listwitheventswithnoerror.append(sample)
           print(index)
   if len(listwitheventswithnoerror) > len(listwitheventswitherror):
       break

testlist = []
for index, sequence in enumerate(listwitheventswitherror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist.append(templist)
    
testlist2 = []
for index, sequence in enumerate(listwitheventswithnoerror):
    templist = []
    for array in sequence:
        array = array[1:]        
        array = np.delete(array, [1])
        templist.append(array)
    testlist2.append(templist)

listwitheventswitherror = testlist
listwitheventswithnoerror = testlist2

listwitheventswitherror[0]
listwitheventswithnoerror[0]


         
#%% Deleting empty arrays and adding labels
import random
    
tagerror = [1]
tagnoerror = [0]

## adding the label as the last entry of each array.

addedarraywitherror = []
addedarraywithnoerror = []

for index, array in enumerate(listwitheventswitherror):
    added = np.append(array, tagerror)
    addedarraywitherror.append(added)
addedarraywitherror = np.asarray(addedarraywitherror)

for index, array in enumerate(listwitheventswithnoerror):
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
#model.load_weights("best_model_randomN5.h5")

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss'),
             ModelCheckpoint(filepath='best_model_randomN5.h5', monitor='val_loss', save_best_only=True)]
model.summary()

filepath = 'best_model_randomN5.h5'
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
plt.savefig('accuracy_window5.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_window5.png')
plt.show()

predictions = model.predict(X_test, verbose=1)





