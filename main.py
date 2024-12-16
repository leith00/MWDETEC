#local
from trainings_routine import *
from objective import *
from utils import *
from eval import *
import settings


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
import xgboost as xgb



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#MODE = BODMAS,EMBER, CONCEPT_DRIFT

if settings.MODE == "BODMAS":
    Dataset = "BODMAS"
    X_train, X_test, y_train, y_test, X_val, y_val, Xcv,ycv = load_data(settings.PATH_BODMAS)
#EMBER
if settings.MODE == "EMBER":
    Dataset = "EMBER"
    X_train, X_test, y_train, y_test, X_val, y_val, Xcv, ycv = load_ember()

#Erzeuge ID für experimentenSammlung
ExperimentSuite = uuid.uuid4()




##################Random Forrest##################

categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')

preprocessor = ColumnTransformer(
    transformers=[
        ('num',preprocessing.MinMaxScaler(), numerical_features)
        #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# SMOTE anwenden (nach Transformation der Features)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)


#scaler = preprocessing.MinMaxScaler()
clf = RandomForestClassifier(n_estimators=1000, max_features=100, max_samples=2000)
expRF= MwExperiment("Random Forest-BS",[
    ('classifier',clf),  
], [0.2, 0.02,42],X_train_resampled, X_test_transformed, y_train_resampled, y_test)
##baseline_model_fit##
expRF.fit()
print(expRF.score())
expRF.log('dataset_name', 'BODMAS')
expRF.log('percentage_of_split_random_seed', [0.2, 0.02,42])
##erster eval accuracy##



##Model speichern###TODO nur bestes Model speichern unter utils  
#with open ('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Models/modelrf', 'wb') as f:
#    pickle.dump(expRF,f)


##eval Baseline##
###eval(expRF,Dataset+"baseline_modelRF.pdf")


#default clf score speicghern 
#expRF.log('base_model_acc', acc)
expRF_opt = deepcopy(expRF)
expRF_opt.name = "Random Forest-opt"
##Hyperparameter search##
expRF_opt.optimize(lambda trial: cross_val_score(objectiveRF(trial,expRF_opt.pipe),Xcv, ycv, cv=3).mean())
expRF_opt.best_model()


#expRF.save()

##eval best_model##
eval(expRF_opt,Dataset+"best_modelRF.pdf")
evalstudy(expRF_opt,Dataset+"RF-optunahistory")







##################XGBoost ##################
categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')
preprocessor = ColumnTransformer(
    transformers=[
        ('num',preprocessing.StandardScaler(), numerical_features)
        #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# SMOTE anwenden (nach Transformation der Features)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)



expXGB = MwExperiment("XGBoost Baseline",[
    ("classifier", clf)
], [0.2, 0.02,42],X_train_resampled, X_test_transformed, y_train_resampled, y_test)
##baseline_model_fit##
expXGB.fit()

#hyperparameter search 
eval(expXGB, Dataset+"baseline_modelXGB.pdf")
expXGB_opt= deepcopy(expXGB)
expXGB_opt.name = "XGBoost-opt"

expXGB_opt.optimize(lambda trial: cross_val_score(objectiveModelXGB(trial,expXGB_opt.pipe), Xcv, ycv, cv=3).mean())
expXGB_opt.best_model()


eval(expXGB_opt,Dataset+"best_modelXGB.pdf")
evalstudy(expXGB_opt,Dataset+"XGB-optunahistory")







##################Multi Layer Percepton ##################
categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')


preprocessor = ColumnTransformer(
    transformers=[
        ('num',preprocessing.StandardScaler(), numerical_features)
        #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# SMOTE anwenden (nach Transformation der Features)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
print("Daten vorbereitet")
#aus utils rein 
def mlp_clf(input_shape=(X_train_resampled.shape[1],)):
    inx = Input(input_shape)
    x = layers.Dense(512, activation='relu')(inx)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation=None)(x)
    model = Model(inx, x)
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


expMLP = MwExperimentDL("MLP Baseline",[
    ('classifier', KerasClassifier(model=mlp_clf, verbose=0))     
], [0.2, 0.02,42], X_train_resampled, X_test_transformed, y_train_resampled, y_test)

mlp = mlp_clf()
mlp.summary()
print("MLP initialisiert")
#muss anpassen
#expMLP.compile()
expMLP.fit()
print("fertig trainiert")
expMLP.evaluate()
expMLP.predict()

eval(expMLP, Dataset+"baseline_modelMLP.pdf")
expMLP_opt= deepcopy(expMLP)
expMLP_opt.name = "MLP-opt"

expMLP_opt.optimize(lambda trial: cross_val_score(objectiveModelMLP(trial,expMLP_opt.pipe,(expMLP_opt.X_train.shape[1],)), Xcv, ycv, cv=3).mean())
expMLP_opt.best_modelDL()



eval(expMLP_opt,Dataset+"best_modelMLP.pdf")
evalstudy(expMLP_opt, Dataset+"MLP-optunahistory")








##################Support Vector Machine##################

categorical_features, numerical_features = get_num_cat_features('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Data/raw_data/bodmas.parquet')
#scaling beobachten
preprocessor = ColumnTransformer(
    transformers=[
        ('num',preprocessing.StandardScaler(), numerical_features)
        #('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

# Wandle die Trainingsdaten mit dem Preprocessor um
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# SMOTE anwenden (nach Transformation der Features)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)


pca = PCA(n_components=50)
clf = svm.LinearSVC(dual=False, max_iter=2000, tol=1e-3)
#cali hinzufügen
cali_clf = CalibratedClassifierCV(estimator=clf, cv=3)


expSVM = MwExperiment("SVM Baseline",[
    ('pca',pca),
    ('classifier',cali_clf),  
], [0.2, 0.02,42],X_train_resampled, X_test_transformed, y_train_resampled, y_test)
##baseline_model_fit##
expSVM.fit()
#acc=pipeSVM.score(X_test,y_test)
#print(acc)
#hyperparameter search 
eval(expSVM, Dataset+"baseline_modelSVM.pdf")

expSVM_opt= deepcopy(expSVM)
expSVM_opt.name = "SVM-opt"


expSVM_opt.optimize(lambda trial: cross_val_score(objectiveModelSVM(trial,expSVM_opt.pipe), Xcv, ycv, cv=3).mean())

expSVM_opt.best_modelSVM()

eval(expSVM_opt,Dataset+"best_modelSVM.pdf")
evalstudy(expSVM_opt,Dataset+"SVM-optunahistory")





resulttable([expRF, expXGB, expMLP, expSVM],Dataset+"BS"+str(ExperimentSuite),"Baseline models - supervised" + "Random_state: 42 ,Train-test-split = 80/20 ")
#resulttable([expRF_opt, expXGB_opt,expMLP_opt, expSVM_opt],Dataset+"opt"+str(ExperimentSuite),"Optimal models - supervised" + "Random_state: 42,Train-test-split = 80/20 ")

roc([expRF, expXGB, expMLP, expSVM],Dataset+"roc_Baseline"+str(ExperimentSuite)+".pdf")
#roc([expRF_opt, expXGB_opt,expMLP_opt, expSVM_opt],Dataset+"roc_opt"+str(ExperimentSuite)+".pdf")



'''
names = ["Random Forest-BS","Random Forest-opt","XGBoost Baseline","XGBoost-opt","MLP Baseline","MLP-opt","SVM Baseline","SVM-opt"]


score_list=[]
for n in names:
    with open('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Predictions/'+ n, "rb") as f:
        score = pickle.load(f)
        score_list.append(score)
        f.close()

roc_(score_list,y_test,Dataset+"roc_opt"+str(ExperimentSuite)+".pdf")
'''



