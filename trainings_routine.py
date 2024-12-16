#local
from utils import *


import numpy as np
import pandas as pd 
import uuid
import pickle
import os.path
import json


import optuna 
from sklearn.pipeline import Pipeline

class MwExperiment:
    def __init__(self, name, steps,split,X_train,X_test,y_train,y_test):
        self.id = uuid.uuid4()                    
        self.name = name
        self.pipe = Pipeline(steps)
        self.split = split
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.resultlog = dict()
        self.resultlog['run_id'] = str(self.id)


    def fit(self):
        self.pipe.fit(self.X_train,self.y_train)

    def score(self):
        return self.pipe.score(self.X_test,self.y_test) 
    
    
    def predict(self):
        return self.pipe.predict(self.X_test)
    
    def predict_proba(self):
        return self.pipe.predict_proba(self.X_test)
    
    def decision_function(self):
        return self.pipe.decision_function(self.X_test)

    def optimize(self,objective):
        self.study = optuna.create_study(direction = "maximize", pruner=optuna.pruners.MedianPruner())
        self.study.optimize(objective, n_trials = 50, n_jobs=6)
        print("suche nehme besten Trial raus")
        trial = self.study.best_trial
        print("Best Score: ", trial.value)
        print("Best Params: ")
        #self.resultlog['best_score_acc'] = trial.value
        for key, value in trial.params.items():
            print("  {}: {}".format(key, value))
            #self.resultlog[key] = value 
        print("optimize fertig")


    def best_model(self):
        best_params = self.study.best_trial.params
        prefix = "classifier__" 
        best_params = {prefix + key: value for key, value in best_params.items()} 
        print(best_params)
        self.pipe.set_params(**best_params)
        print("trainiere bestmodel")
        self.fit()
        print("Best model Accuracy: " +str(self.score()))

    def best_modelSVM(self):
        best_params = self.study.best_trial.params
        cali_clf = self.pipe['classifier']
        svm = cali_clf.estimator
        C = best_params.get('C')
        svm.set_params(C=C)
        self.fit()
    

 
    def log(self,key, value):
        self.resultlog[key] = value

    def save(self):
        file = open('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Output/Experiments/exp'+str(self.id), "wb")
        pickle.dump(self, file)
        file.close()
        
        #serializing json
        json_object = json.dumps(self.resultlog)
        with open ('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Output/Experiments/exp'+str(self.id)+'.json', "w") as outfile:
            outfile.write(json_object)
 
    def load(self,id):
        file = open('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Output/Experiments/exp'+id, "rb")
        loaded_obj = pickle.load(file)
        self.__dict__.update(loaded_obj.__dict__)                 
        file.close()    

class MwExperimentAnomalie(MwExperiment):
    def __init__(self, name, steps, split, X_train, X_test, y_train, y_test):
        super().__init__(name, steps, split, X_train, X_test, y_train, y_test)
    
    def fit(self):
        self.pipe.fit(self.X_train)


class MwExperimentDL(MwExperiment):
    def __init__(self,name,steps, split, X_train, X_test, y_train, y_test):
        super().__init__(name,steps,split, X_train, X_test, y_train, y_test)

    def compile(self):
        self.pipe['classifier'].compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    def fit(self):
        #Sklearn fit 
        for step in self.pipe.steps[:-1]:           #alle steps statt dem letzen (classifier) 
            self.pipe[step[0]].fit(self.X_train,self.y_train)
        #DL fit
        self.pipe['classifier'].fit(self.X_train, 
                    self.y_train, 
                    epochs=21, 
                    batch_size=16, 
                    validation_split=0.2)
    
    def evaluate(self):
        test_loss, test_acc = self.pipe['classifier'].model_.evaluate(self.X_test, self.y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_loss, test_acc

    def predict(self):
        y_pred = self.pipe.predict(self.X_test)
        return y_pred
    
    

    def best_modelDL(self):
        best_params = self.study.best_trial.params 
        print(best_params)
        model = mlp_clf((self.X_train.shape[1],),best_params['units'],best_params['dropout_rate'])
        model.fit(self.X_train, 
                    self.y_train, 
                    epochs=best_params['epochs'], 
                    batch_size=best_params['batch_size'], 
                    validation_split=0.2)
        print("Best model Accuracy: " +str(self.score()))
       
    
        


class MwExperimentAE(MwExperiment):
    def __init__(self,name,steps, split, X_train, X_test, y_train, y_test):
        super().__init__(name,steps,split, X_train, X_test, y_train, y_test)


    def compile(self):
        self.pipe['classifier'].compile(optimizer='adam', 
              loss='mse', 
              metrics=['accuracy'])

    def fit(self):
        #Sklearn fit 
        for step in self.pipe.steps[:-1]:          
            self.pipe[step[0]].fit(self.X_train,self.y_train)
        #DL fit
        self.pipe['classifier'].fit(self.X_train, 
                    self.X_train, 
                    epochs=21, 
                    batch_size=128, #16
                    validation_data=(self.X_test, self.X_test))  

    def predict(self):
        y_pred = self.pipe.predict(self.X_test)
        return y_pred

