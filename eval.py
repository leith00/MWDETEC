import settings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import optuna 
import plotly.io
import pickle

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn_evaluation import plot

anomalie = ["Autoencoder Baseline", "Autoencoder OPT", "Isolation Random Forest Baseline", "Isolation Random Forest OPT", "OC-SVM Baseline", "OC-SVM OPT"]


def prep_for_score_sk(experiment):
    if experiment.name  not in anomalie:
        y_pred = experiment.predict()
        y_score = experiment.predict_proba()
        #print(experiment.y_test.shape)
        #print(y_score[:,1].shape)
        roc_auc= roc_auc_score(experiment.y_test,y_score[:,1])       # brauche von dem 2 dim array die 2te spalte
    else:
        #unsupervised
        y_score = experiment.decision_function()
        y_pred = experiment.predict()                     #1 beningn , -1 für malicous    Labels gleich setzten
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        roc_auc= roc_auc_score(experiment.y_test,y_score)    
    
    acc = accuracy_score(experiment.y_test, y_pred)

    return y_pred, y_score, acc , roc_auc

def prep_for_score_tf(experiment):
    y_score = experiment.predict()
    y_score_ = np.array([[float(x), float(1-x)]for x in y_score.flatten()])
    y_pred = (y_score > 0.5).astype("int32").astype("float32").flatten()        #set threshold to get probability and cast boolean to int 
    _, acc = experiment.evaluate()
    roc_auc= roc_auc_score(experiment.y_test,y_score_[:,1])

    return y_pred, y_score_, acc, roc_auc

def prep_for_score_AE(experiment):
    reconstructions = experiment.predict()
    # Rekonstruktionsfehler berechnen 
    reconstruction_errors = np.mean((experiment.X_test - reconstructions) ** 2, axis=1)

    # Schwellenwert festlegen 
    fpr, tpr, thresholds = roc_curve(experiment.y_test, reconstruction_errors)
    optimal_idx = np.argmax(tpr - fpr)
    opt_threshold= thresholds[optimal_idx]
    print(f"opt threshold: {opt_threshold:.4f}")
    y_pred = (reconstruction_errors>=opt_threshold).astype(int)
    y_score = reconstruction_errors

    acc = accuracy_score(experiment.y_test, y_pred)
    roc_auc= roc_auc_score(experiment.y_test,y_score) 
    
    return y_pred, y_score, acc, roc_auc

def get_scores(experiment):
    if experiment.name in ["Autoencoder Baseline", "Autoencoder OPT"]:
        y_pred, y_score, acc, roc_auc = prep_for_score_AE(experiment)        
    #print(type(experiment).__name__)
    elif type(experiment).__name__=="MwExperimentDL":
        y_pred, y_score, acc, roc_auc = prep_for_score_tf(experiment)
    else:
        y_pred, y_score, acc, roc_auc = prep_for_score_sk(experiment)

    y_true = experiment.y_test
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    #f1, precision, recall, roc_auc

    #with open ('/Users/leith/Documents/IT/B.s/Code/mwdfinal_backUp_15.10/Predictions/'+ experiment.name, 'wb') as f:
    #    pickle.dump((y_pred,y_score,roc_auc),f)
    return y_pred,y_true,y_score, tpr,fpr, acc, f1_score(y_true,y_pred), precision_score(y_true,y_pred), recall_score(y_true, y_pred), roc_auc


def eval(experiment,pdf_filename):

    # Create a PDF file to save the plots
    with PdfPages(pdf_filename) as pdf_pages:            #with sorgt dafür das ws in context manager geöfnnet wird unD yleanup action macht   
        y_pred, _, y_score, *_ = get_scores(experiment) 

        ################
        #confusion matrix
        #cf_matrix = confusion_matrix(experiment.y_test, y_pred)
        #plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt='0.2%')
        if settings.MODE == "BODMAS":
            plot.ConfusionMatrix.from_raw_data(experiment.y_test, y_pred)
            pdf_pages.savefig()#save this plot to the pdf
        #classification report
        target_names = ['Non-malicious','malicious']
        plot.ClassificationReport.from_raw_data(experiment.y_test, y_pred, target_names=target_names)
        pdf_pages.savefig()    


        # plot precision recall curve
        plot.precision_recall(experiment.y_test, y_pred)
        pdf_pages.savefig()    

        #ROC
        plot.ROC.from_raw_data(experiment.y_test, y_score)
        pdf_pages.savefig()
        plt.clf()
        plt.close('all')


def roc(experiment_list,pdf_filename):
    with PdfPages(pdf_filename) as pdf_pages: 

        plt.figure(0).clf()
        for exp in experiment_list:
            y_pred,y_true,_, _, _, _, _, _, _, roc_auc=get_scores(exp)
                    
            fpr,tpr,_= roc_curve(y_true,y_pred)
            auc = roc_auc
            plt.plot(fpr,tpr,label=exp.name+", AUC="+str(auc))
        plt.legend()
        pdf_pages.savefig()
        plt.close('all')

def roc_(scores_list,y_true,pdf_filename,name):
    with PdfPages(pdf_filename) as pdf_pages: 

        plt.figure(0).clf()
        for score in scores_list:
            y_pred = score[0]
            auc = score[2]
                    
            fpr,tpr,_= roc_curve(y_true,y_pred)
            plt.plot(fpr,tpr,label=name+", AUC="+str(auc))
        plt.legend()
        pdf_pages.savefig()
        plt.close('all')


def evalstudy(experiment,name):
    # Create a PDF file to save the plots
    #with PdfPages(pdf_filename) as pdf_pages: 

    #Optuna optimization history
    fig = optuna.visualization.plot_optimization_history(experiment.study)
    plotly.io.show(fig)
    plotly.io.write_image(fig, 'study'+str(name)+'.pdf',format='pdf')
        #pdf_pages.savefig()        



def resulttable(exp_list, id, description):

    metrics = ["TPR", "FPR", "Accuracy", "F1", "Precision", "Recall", "ROC AUC"]
    results = {}
    #alle scores pro exp
    for exp in exp_list:
        _,_,_, tpr,fpr, acc, f1, precision, recall, roc_auc = get_scores(exp)
        #print("TN:  "+ str(tn) + "  FP:  "+ str(fp)+ "  FN:  "+str(fn)+"  TP:  "+str(tp)+"  acc:  "+str(acc)+"  F1:  "+str(f1)+"  Precision:  "+str(precision)+"  recall:  "+str(recall)+"  roc_auc:  "+str(roc_auc))
        results[exp.name] = [tpr, fpr, acc, f1, precision, recall, roc_auc]

    #in Pandas df
    df = pd.DataFrame(results, index=metrics)
    with open("results_table"+id+".csv", 'w') as f :
        f.write(f"# {description}\n")
        print(df)
        df.to_csv(f, index=True)

