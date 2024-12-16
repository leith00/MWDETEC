from utils import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

import setuptools.dist
import tensorflow as tf
import keras
from keras import Input, Model, layers, losses, optimizers, callbacks
from scikeras.wrappers import KerasClassifier




def objectiveRF(trial, pipe):

    max_depth = trial.suggest_int("max_depth", 2, 64, log=True)
    max_samples = trial.suggest_float("max_samples", 0.2, 1)
    max_features = trial.suggest_categorical('max_features', [50, 100])

    pipe.set_params(classifier__max_depth=max_depth)
    pipe.set_params(classifier__max_samples=max_samples)
    pipe.set_params(classifier__max_features=max_features)
    return pipe


def objectiveModelSVM(trial, pipe):

    C = trial.suggest_loguniform("C", 10**-3, 10**3)

    cali_clf = pipe['classifier']
    svm = cali_clf.estimator

    svm.set_params(C=C)
    return pipe




def objectiveModelXGB(trial,pipe):
  
    pipe.set_params(classifier__tree_method=trial.suggest_categorical('tree_method', ['approx', 'hist']))
    pipe.set_params(classifier__max_depth=trial.suggest_int('max_depth', 2, 20))      
    pipe.set_params(classifier__min_child_weight=trial.suggest_int('min_child_weight', 1, 250))                
    pipe.set_params(classifier__subsample=trial.suggest_float('subsample', 0.1, 1.0))
    pipe.set_params(classifier__colsample_bynode=trial.suggest_float('colsample_bynode', 0.1, 1.0))
    pipe.set_params(classifier__reg_lambda=trial.suggest_float('reg_lambda', 0.001, 25, log=True))
    pipe.set_params(classifier__learning_rate=0.3)  

 
    return pipe
              
    




def objectiveModelMLP(trial, pipe, shape):
    
    units = trial.suggest_int("units", 32, 2048, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    epochs = trial.suggest_int("epochs", 10, 50, step=10)  

    pipe.set_params(classifier__model=lambda: mlp_clf(shape,units, dropout_rate),
                    classifier__batch_size=batch_size,
                    classifier__epochs=epochs)

    return pipe
                
    
        


def objectiveModelIRF(trial,pipe):
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    max_features = int(max_features * 2300) 
    pipe.set_params(classifier__n_estimators = trial.suggest_int(name="n_estimators", low=100, high=2000, step=100))
    pipe.set_params(classifier__max_features =  max_features)


    return pipe

#kernel aprox oc svm 