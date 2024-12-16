#local
import settings

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import setuptools.dist
import tensorflow as tf
import keras
from keras import Input, Model, layers, losses, optimizers, callbacks
from scikeras.wrappers import KerasClassifier



def datasplit(X,y,test_size):
    X_, X_test, y_, y_test = train_test_split(X,y, test_size=test_size, random_state=42)#70train/10val/20test
    X_train, X_val, y_train, y_val = train_test_split(X_,y_, test_size=0.125, random_state=42) #X_train__, y_train__ verwerfen nur val ist wichtig
    Xcv,X_,ycv,Y_ = train_test_split(X_train,y_train, train_size=20000, random_state=42)
    return X_train, X_test, y_train, y_test, X_val, y_val, Xcv,ycv


def load_data(path,test_size=0.2):
    df = pd.read_parquet (path)
    #für gesamtes Dataset zeile 27 einkommentieren
    X = df.iloc[:,:2381]    
    y = df.iloc[:,2381]
    return datasplit(X,y,test_size)


def load_ember(test_size=0.2):
    df_train = pd.read_parquet(settings.PATH_EMBER_TRAIN)
    df_test = pd.read_parquet(settings.PATH_EMBER_TEST)

    df_train = df_train[df_train['Label'].isin([0,1])]

    df = pd.concat(objs=[df_train, df_test], axis=0, copy=True, sort=False, ignore_index=True)
    df = df.drop_duplicates()

    df_sample = df.sample(n=150000,random_state=42)

    X = df_sample.drop('Label', axis=1)
    y = df_sample['Label']
    #X = X.sample(n=150000, random_state=42)#15 0 000
    #y = y.sample(n=150000, random_state=42)
    return datasplit(X,y,test_size)



def get_num_cat_features(path):
    df = pd.read_parquet(path)
    df = df.drop(columns=['Label','sha256'])

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    ##print(categorical_features)
    # Numerische Features (Datentypen wie 'int64', 'float64')
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    
    return categorical_features, numerical_features


def mlp_clf(shape, units, dropout_rate):
    inx = Input(shape)
    x = layers.Dense(units, activation='relu')(inx)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(int(units/2), activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(int(units/4), activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation=None)(x)
    model = Model(inx, x)
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])  
    
    return model 