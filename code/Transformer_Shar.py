#!/usr/bin/env python
# coding: utf-8

# This script-that-used-to-be-a-notebook reads in Meg's black hole binary/quasar data and trains a transformer-based binary classifier on that data.
#make sure to change the path for the model to get saved to, which is at the very end of the notebook rn


from astropy.table import Table, Column
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

import os


#change to your path for your downloaded dataset, whole notebook uses this
# lc_file = '/content/drive/MyDrive/hackathon_binaries_meg/Hackathon_2024_lcs_DDF.hdf5'

varying_binaries = '/home/3155/meg/data/Hackathon_2024_varyingbinaries_small.hdf5' #for one file

folder_path = "/home/3155/meg/data/" #if you want to read in all the data from a folder



#looking at file structure
f = h5py.File(varying_binaries, 'r')

print(f.keys())


def read_in_data(filepath, num_desired=11000):
  '''
  this function takes the path to a synthesized hdf5 data file
  and returns the data (format: dataframe) and corresponding labels (format: array)
  num_desired is a way to shorten the dataframe if you want to use less data
  '''
  f = h5py.File(filepath, 'r')
  for i, key in enumerate(f.keys()):
    #let's get the truth table from astropy table
    #there must be another way to do this but i don't know it and this is fast
    data = Table.read(filepath, format='hdf5', path=key) #read in the dataset in the file

    if i==0:
      truths = data.meta['truths']
      df_pd = pd.read_hdf(filepath, key).T #god i wish i could turn those error statements off
      df_pd = df_pd.drop(labels="time", axis=0)
    else:
      if len(df_pd) >= num_desired:
            df_pd = df_pd[:num_desired]
            truths = truths[:num_desired]
            return df_pd, truths
      truths_new = data.meta['truths']
      truths = np.concatenate((truths, truths_new))
      df_pd_new = pd.read_hdf(filepath, key).T #god i wish i could turn those error statements off
      df_pd_new = df_pd_new.drop(labels="time", axis=0)
      df_pd = pd.concat([df_pd, df_pd_new], ignore_index=True)

    
  return df_pd, truths



def read_in_data_folder(folderpath):
    '''
    this function takes the path to folder of synthesized hdf5 data files
    and returns one dataframe of all the time series and an array of corresponding labels
    '''
    for i, file in enumerate(os.listdir(folderpath)):
        data_filename = os.fsdecode(file)
        data_filename_full = folderpath + data_filename
        print(data_filename)
        if i==0:
            df_pd, truths = read_in_data(data_filename_full)
        else:
            df_pd_new, truths_new = read_in_data(data_filename_full)
            df_pd = pd.concat([df_pd, df_pd_new], ignore_index=True)
            truths = np.concatenate((truths, truths_new))
    return df_pd, truths
    


#use the below if you want one data file
# df_pd, truths = read_in_data(varying_binaries)

#use the below if you want a whole folder of files
df_pd, truths = read_in_data_folder(folder_path)


df_pd.shape, truths.shape #checking shapes are as expected


df_pd.values[1002,:] #making sure that this isn't a time row




#i'm gonna do a few things to clean up the data...

#time is the axis label, so i'm removing it from the main dataframe
# df_pd = df_pd.drop(labels="time", axis=0)

#turning all of the missing values into nan's and then removing those times
df_pd = df_pd[df_pd>-900]
df_cut = df_pd.dropna(axis=1)
#note - will need a more sophisticated method for this if the gaps in
#other datasets are not all at the same time

#also going to minmax normalize because the transformer will learn faster on that
def minmax_normalize(df):
    normalized_df = df.apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)
    return normalized_df

normalized_df = minmax_normalize(df_cut)


#now splitting up the data into a training and testing set

data_train, data_test, label_train, label_test  = train_test_split(
     normalized_df, truths, test_size=0.3, random_state=4)


#keras likes for there to be an extra dimension

def add_dim(X, swap=False):
    vals = X.values
    print(type(vals))
    reshaped = vals.reshape(vals.shape[0], vals.shape[1], 1)
    if swap:
        reshaped = np.swapaxes(reshaped, 1, 2)
    return reshaped

Xtrn = add_dim(data_train, swap=True)
Xtst = add_dim(data_test, swap=True)



input_shape = Xtrn.shape[1:]
num_wvls = Xtrn.shape[2]
#num_wvls is the number of data points in each time series



#transformer architecture

def build_model(input_shape, num_wvls, num_classes, num_transformer_blocks):
    '''
    input_shape is Xtrn.shape[1:]
    num_wvls is the number of data points in each time series: Xtrn.shape[2]
    num_classes is two because this is a binary classification
    num_transformer_blocks is a parameter that you can play with 
    - it's basically how large the transformer part is.
    there are some more things that you can change that are not in the function call (i guess maybe they should be oops)
    there are regularizers, like keras.regularizers.l2(0.01),
    and making that number larger will make the model less prone to overfitting but also harder for it to learn
    you can also adjust the num_heads and key_dim - those are transformer-specific parameters
    and i have not played around with what changes when you adjust them!
    i also have a couple dense layers near the end commented out, those may or may not be useful
    '''
    inputs = layers.Input(shape=input_shape)

    x = inputs
    # Create multiple transformer blocks
    for _ in range(num_transformer_blocks):
        x0 = layers.MultiHeadAttention(num_heads=32, key_dim=128)(x, x) #tried 16 heads and key_dim=64
        x0 = layers.Add()([x, x0])
        x0 = layers.LayerNormalization()(x0)
        x1 = layers.Conv1D(filters=2048, kernel_size=1, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x0) 
        x1 = layers.Conv1D(filters=num_wvls, kernel_size=1, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x1)
        x1 = layers.Add()([x0, x1])
        x1 = layers.LayerNormalization()(x1)
        x = x1  # Update x for the next iteration
        
    #htis is the end of the transformer part, now we work on the classification part
    #the transformer learns info about the time series and the a normal densely
    #connected NN does the classification
    x = layers.GlobalMaxPooling1D(data_format="channels_last")(x) #this used to be x1
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model



#building the model
model = build_model(input_shape, num_wvls, num_classes=2, num_transformer_blocks=4)
model.summary()



#have not messed around with changing these
loss = losses.BinaryCrossentropy()
acc = metrics.BinaryAccuracy(name="ba")
opt = optimizers.Nadam(learning_rate=1e-5)
model.compile(loss=loss, optimizer=opt, metrics=[acc])



#early stopping callback
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.01, patience=5, restore_best_weights=True)
# This callback will stop the training when there is no improvement in
# the loss for five consecutive epochs.
#also will restore the best weights instead of the most recent


#training the model

history = model.fit(
    Xtrn,
    label_train,
    batch_size=15, #can adjust this, changes amount of memory used
    callbacks=earlystopping,
    epochs=30, #increase this if you can, it will automatically end when it stops improving because of the early stopping callback
    validation_split=0.1,
    verbose=1
)


#saving the model and outputs - make sure to adjust these paths

model.save("NN_apr21")
np.save("/home/3155/meg/history_apr21/loss_history", history.history["loss"])
np.save("/home/3155/meg/history_apr21/val_loss_history", history.history["val_loss"])
np.save("/home/3155/meg/history_apr21/accuracy_history", history.history["ba"])
np.save("/home/3155/meg/history_apr21/val_accuracy_history", history.history["val_ba"])
