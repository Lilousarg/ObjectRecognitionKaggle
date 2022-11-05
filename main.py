# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:39:36 2022

@author: lilou
"""
#%%
import json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from seaborn import countplot
from PIL import Image
from keras.layers import Input, Conv2D, Activation, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras import regularizers
from keras import optimizers
from keras import Sequential
#%%
def plot_curves(history):
  plt.figure(figsize=(16, 6))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Training', 'Validation'])
  plt.title('Loss')

  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Training', 'Validation'])
  plt.title('Accuracy')
  
def define_model_l2(lambda_):
  model = Sequential()
  model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_), input_shape=(400, 400, 3)))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_)))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_)))
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(lambda_)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(lambda_)))

  model.add(Flatten())
  model.add(Dense(92, activation='softmax'))
  return(model)

def define_model_dropout(dropout_rate = 0):
  model = Sequential()
  model.add(Conv2D(32, (3,3), padding='same', input_shape=(400, 400, 3)))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(dropout_rate))

  model.add(Conv2D(64, (3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(dropout_rate))

  model.add(Conv2D(128, (3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(dropout_rate))
  
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(dropout_rate))

  model.add(Flatten())
  model.add(Dense(92, activation='softmax'))
  return(model)
#%%
train =  pd.read_csv('train.csv')
print(train.head())


#%%
train_new = train.drop([])

#%%
labels = pd.read_csv('labels.csv')
print(labels)
#%%
col = np.zeros((len(train_new), len(labels)))
#%%
labelz = []
for i in range(0, len(labels)):
    labelz.append(str(labels['object'][i])) 
table = pd.DataFrame(data=col, columns= [labelz], index = [train_new.index]) 
print(table)
#%%
for i in range(0,len(train_new)):
    for j in range(0,len(labels)):
        t = "l" + str(j) + " "
        if t in train['labels'].iloc[i]:
            table.iloc[i,j] = 1
 
#%%          
train_test = pd.merge(train_new, table, how='outer', on= train_new.index)

#%%

#%%
from keras_preprocessing import image

img_dir = 'images/images/'
X_datasets = []

for i in range(0,len(train_test)):
    img = image.load_img(img_dir +train_test['image_id'].iloc[i], target_size=(400,400,3))
    img = image.img_to_array(img)
    
    img = img/255.
    X_datasets.append(img)

#%%
X = np.array(X_datasets)
y = train_test.drop(columns=['image_id', 'labels', 'key_0'])


#%%
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=(20), test_size=0.2)

#%%
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=False)
model = define_model_dropout(0.4)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test), batch_size = 64)

#%%
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=False)
model = define_model_l2(0.001)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test), batch_size = 64)

#%%
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(400,400,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation=('relu')))
model.add(Dropout(0.4))
model.add(Dense(128, activation=('relu')))
#model.add(Dropout(0.4))
model.add(Dense(92,activation=('sigmoid')))



#%%
model.summary()

#%%
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

#%%
history = model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test), batch_size = 64)


#%%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=2, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
model.fit(X_train, y_train)

#%%nogo
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

datagen = ImageDataGenerator( # randomly rotate an image from 0 to 90 degrees
                 width_shift_range=0.1,  # horizontally shift an image by a fraction of 0% - 10% (of original width)   
                 height_shift_range=0.1, # vertically shift an image by a fraction of 0% - 10% (of original height)   
                 horizontal_flip=True) # horizontaly flip random 30% of images 

datagen.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=(20), test_size=0.2)

history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    steps_per_epoch=X_train.shape[0]//64, # number of steps per epochs, needs to be specified as we do augmentation
                    epochs=14,
                    validation_data=(X_test, y_test)
                    )

#%% THE SUBMISSION STARTS HERE
test = pd.read_csv('test.csv')
test2 = test.copy(deep=True)
test2 = test2.replace(to_replace='img285.jpg', value='img118.jpg')
test2 = test2.replace(to_replace='img288.jpg', value='img128.jpg')

#%%
X_dataset2 = []
for i in range(0,len(test)):
    img = image.load_img(img_dir +test2['image_id'].iloc[i], target_size=(400,400,3))
    img = image.img_to_array(img)
    
    img = img/255.
    X_dataset2.append(img)

X2 = np.array(X_dataset2)
#%%
val_predictions_test = model.predict(X2)

#%%
val_predictions=['']*len(val_predictions_test)
for i in range(0,len(val_predictions_test)):
    for j in range(0,len(labels)):
        t = "l" + str(j) + " "
        if val_predictions_test[i,j] > 0.499:
            val_predictions[i] = val_predictions[i] + t

#%%
trying = pd.DataFrame({
    'image_id': test['image_id'],
    'labels': val_predictions})
trying['labels']=trying['labels'].str.rstrip()
#%%
submission = trying
submission.to_csv('submission_baseline.csv', index = False)
#%%
