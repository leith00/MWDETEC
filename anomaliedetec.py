#local
import settings
from trainings_routine import *
from objective import *
from utils import *
from eval import *


import pandas as pd
import numpy as np 
import json
import pickle
import uuid
from copy import deepcopy
import setuptools.dist

import tensorflow as tf
import keras
from keras import Input, Model, layers, losses, optimizers, callbacks
from scikeras.wrappers import KerasClassifier

from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve,make_scorer,roc_auc_score, roc_curve
# visualisations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
#matplotlib notebook


if settings.MODE == "BODMAS":
    Dataset = "BODMAS"
    X_train, X_test, y_train, y_test, X_val, y_val, Xcv,ycv = load_data(settings.PATH_BODMAS)
#EMBER
if settings.MODE == "EMBER":
    Dataset = "EMBER"
    X_train, X_test, y_train, y_test, X_val, y_val ,Xcv,ycv= load_ember()

#Erzeuge ID für experimentenSammlung
ExperimentSuite = uuid.uuid4()


#Nur Benign-Daten für das Unsupervised-Training verwenden
#Contamination 10 % malicous mit ins Trainset 
X_train_benign = X_train[y_train == 0]
X_train_malicous = X_train[y_train == 1]
X_train_contamination = X_train_malicous.sample(n=int(len(X_train_benign)*0.1), random_state=42)
X_train_anomalie = pd.concat([X_train_benign, X_train_contamination])

##################Autoencoder##################


categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')

preprocessor = ColumnTransformer(
    transformers=[
        ('num',preprocessing.MinMaxScaler(), numerical_features)
        #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train_benign)
X_test_transformed = preprocessor.transform(X_test)

# SMOTE anwenden (nach Transformation der Features)
##smote = SMOTE(random_state=42)
##X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

#baseline model#

input_dim = X_train_transformed.shape[1]
batch_size = 256
epochs = 3
print(input_dim)

autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    #encoder
    #tf.keras.layers.Dense(input_dim,activation='elu',input_shape=(input_dim,)),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),

    # decode
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(input_dim, activation='linear')    
])
print("finished compile")
autoencoder.compile(optimizer="adam", 
                loss="mse",
                metrics=["acc"])

expAE = MwExperimentAE("Autoencoder Baseline",[
('classifier', autoencoder)
],[0.2, 0.02,42],X_train_transformed, X_test_transformed, X_train_transformed, y_test)

expAE.fit()
# print an overview of our model

autoencoder.summary()

history = autoencoder.fit(
X_train_transformed, X_train_transformed,
shuffle=True,
epochs=epochs,
batch_size=batch_size
)
reconstructions = autoencoder.predict(X_val)
# Rekonstruktionsfehler berechnen 
reconstruction_errors = np.mean((X_val - reconstructions) ** 2, axis=1)

# Schwellenwert festlegen 
fpr, tpr, thresholds = roc_curve(y_val, reconstruction_errors)
optimal_idx = np.argmax(tpr - fpr)
opt_threshold= thresholds[optimal_idx]
print(f"opt threshold: {opt_threshold:.4f}")
y_pred = (reconstruction_errors>=opt_threshold).astype(int)

# Evaluierung des Modells
print("Autoencoder - Klassifikationsbericht:")
print(classification_report(y_val, y_pred))




##################Isolation Forrest##################

categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')

preprocessor = ColumnTransformer(
transformers=[
    ('num',preprocessing.MinMaxScaler(), numerical_features)
    #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train_anomalie)
X_test_transformed = preprocessor.transform(X_test)

clf = IsolationForest(n_estimators=2000, contamination=0.1, random_state=42)


expIRF = MwExperimentAnomalie("Isolation Random Forest Baseline",[
                    ("classifier", clf)
                    ], [0.2, 0.02,42],X_train_transformed, X_test_transformed, X_train_transformed, y_test)


##baseline_model_fit##
expIRF.fit()

#hyperparameter search 
##eval(expIRF, "baseline_modelIRF.pdf")
##roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=False)
##expIRF.optimize(lambda trial: cross_val_score(objectiveModelIRF(trial,expIRF.pipe), expIRF.X_train, expIRF.y_train,scoring=roc_auc_scorer, cv=3).mean())
##expIRF.best_model()

##eval(expIRF,"best_modelXGB.pdf")
##evalstudy(expIRF,"XGB-optunahistory")
expIRF_opt= deepcopy(expIRF)
expIRF_opt.name = "Isolation Random Forest OPT"



##################One-Class Support vector machine##################

categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')

preprocessor = ColumnTransformer(
transformers=[
    ('num',preprocessing.MinMaxScaler(), numerical_features)
    #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train_anomalie)
X_test_transformed = preprocessor.transform(X_test)

pca = PCA(n_components=50)
clf = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.09)


expOC_SVM = MwExperimentAnomalie("OC-SVM Baseline",[
('pca', pca),
('classifier', clf)
], [0.2, 0.02,42],X_train_transformed, X_test_transformed, X_train_transformed, y_test)



##baseline_model_fit##
expOC_SVM.fit()

resulttable([expOC_SVM, expAE, expIRF], Dataset+"Anomalie"+str(ExperimentSuite) ,"Models Unsupervised Anomalie detection" + "Random_state: "  + ",Train-test-split = 80/20 ")
##resulttable([expIRF_opt], ExperimentSuite ,"Best models Unsupervised Anomalie detection" + "Random_state: " + str(seed) + ",Train-test-split = 80/20 ")



#resulttable([expRF, expXGB, expMLP],Dataset+"BS"+str(ExperimentSuite),"Baseline models - supervised" + "Random_state: 42 ,Train-test-split = 80/20 ")
#hyperparameter search 
##eval(expOC_SVM, "baseline_modelOC_SVM.pdf")
##expOC_SVM.optimize(lambda trial: cross_val_score(objectiveModelOC_SVM(trial,expOC_SVM.pipe), expOC_SVM.X_train, expOC_SVM.y_train, cv=3).mean())
##expOC_SVM.best_model()

##eval(expOC_SVM,"best_modelOC_SVM.pdf")
##evalstudy(expOC_SVM,"expOC_SVM-optunahistory")
