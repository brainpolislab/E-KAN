from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif,chi2
#from sklearn_genetic import GASearchCV
from sklearn_genetic import GAFeatureSelectionCV
#from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.cross_decomposition import PLSCanonical
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
#from tpot import TPOTClassifier
import shap

from sklearn.metrics import accuracy_score
import tensorflow as tf
#from autogluon.tabular import TabularDataset, TabularPredictor
import keras 
from keras.losses import BinaryCrossentropy
from tpot import TPOTClassifier
from sklearn.ensemble import VotingClassifier

import subprocess # TO RUN VBEOSA

from scikeras.wrappers import KerasClassifier # TO USE VOTING

from kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import random

def KAN_tr_m(X_train,y_train,X_test,y_test,random_s,m,fil,subs,k1,k2,N):
    import random
    def train_K(dataset,l_1,sampled):
        model = KAN(width=[l_1,1,2], grid=6, k=3)
        def train_acc():
            return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

        def test_acc():
            return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

        #results = model.train(dataset, opt="LBFGS", steps=50, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(),lamb=00.9)

        results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
        return model


    # SAMPLE 1 0-->9
    #  SAMPLE 10-->77
    # SAMPLE 77-->:


    # Example array
    array = X_train.columns

    # List to store all sampled arrays
    sampled_arrays = []

    '''
    # Loop to sample elements until fewer than 20 elements remain in the array
    while len(array) >= 20:
        # Randomly sample 20 elements from the array without replacement
        sampled_elements = random.sample(sorted(array), 20)
        sampled_arrays.append(sampled_elements)
    
        # Remove the sampled elements from the original array
        array = [element for element in array if element not in sampled_elements]

    '''
    # List of subgroups: SUBSTANCES, STRUCTURAL, FUNCTIONAL
    if random_s:
        subgroups = [X_train.columns]
    else:
        if subs==True:
            subgroups = [X_train.columns[0:9], X_train.columns[9:77], X_train.columns[77:]]  # Add more subgroups if needed
        if subs==False:
            subgroups = [X_train.columns[0:69], X_train.columns[69:]]

    # List to hold sampled arrays
    sampled_arrays = []
        
    for subgroup in subgroups:
        sample_from_subgroup(subgroup, sampled_arrays)

    # Output the sampled arrays
    #for i, sampled in enumerate(sampled_arrays, 1):
        #print(f"Sample {i}: {sampled}")

    def divide_names(names, N): # ROUND ROBIN
        # Initialize subgroups
        subgroups = [[] for _ in range(N)]
    
        # Distribute names
        for i, name in enumerate(names):
            subgroup_index = i % N
            subgroups[subgroup_index].append(name)
    
        return subgroups
    
    # Divide names into subgroups
    sampled_arrays = divide_names(array, N)


    preds_tr=list()
    preds_te=list()

    # TRAIN KAN ENSEMBLE
    models=[]
    Filt_1=[]
    Filt_2=[]
    for i, sampled in enumerate(sampled_arrays, 1):
        #subgroup = list(subgroup) 
        #print(sampled)
        #extract sampled columns
        X_train_s=X_train[sampled].copy()
        X_test_s=X_test[sampled].copy()
        #extract sampled columns
        X_train_s=X_train[sampled].copy()
        X_test_s=X_test[sampled].copy()
        X_train_s=pd.DataFrame(X_train_s)
        X_test_s=pd.DataFrame(X_test_s)
        if fil==True:
            filt=SelectKBest(f_classif, k=k1).fit(X_train_s, y_train) #k1=10
            X_train_s=filt.transform(X_train_s)
            X_test_s=filt.transform(X_test_s)
            names = filt.get_support(indices=True)
            Filt_1.append(names)
            X_train_s=pd.DataFrame(X_train_s)
            X_test_s=pd.DataFrame(X_test_s)
        dataset= {}
        if m=='KAN':
            #prepare dataset
            dataset= {}
            dataset['train_input'] = torch.from_numpy(X_train_s.values)
            dataset['test_input'] = torch.from_numpy(X_test_s.values)
            dataset['train_label'] = torch.from_numpy(y_train.values[:,0])
            dataset['test_label'] = torch.from_numpy(y_test.values[:,0])

            #train base models
            model=train_K(dataset,dataset['test_input'].shape[1],sampled)
            models.append(model)

            pred=torch.argmax(model(dataset['train_input']), dim=1).numpy().transpose()
            preds_tr.append(pred)
            pred=torch.argmax(model(dataset['test_input']), dim=1).numpy().transpose()
            preds_te.append(pred)
        # THE BASE LEARNER CAN ALSO BE SWITCHED TO OTHERS BUT WE HAVEN'T PRESENTED THIS FEATURE IN THE ARTICLE 
        if m=='SVM': 
            model=svm.SVC()
            model=model.fit(X_train_s,y_train)
            models.append(model)
            pred=model.predict(X_train_s)
            preds_tr.append(pred)
            #print(pred)
            pred=model.predict(X_test_s)
            preds_te.append(pred)
        if m=='MLP':
            model=MLPClassifier(hidden_layer_sizes=(1,2))
            model=model.fit(X_train_s,y_train)
            
            models.append(model)
            pred=model.predict(X_train_s)
            preds_tr.append(pred)
            pred=model.predict(X_test_s)
            preds_te.append(pred)

        if m=='KAN_1':
            X_train=pd.DataFrame(X_train)
            X_test=pd.DataFrame(X_test)
            #filt=SelectKBest(f_classif, k=10).fit(X_train, y_train)
            #X_train=filt.transform(X_train)
            #X_test=filt.transform(X_test)
            #prepare dataset
            dataset= {}
            dataset['train_input'] = torch.from_numpy(X_train.values)
            dataset['test_input'] = torch.from_numpy(X_test.values)
            dataset['train_label'] = torch.from_numpy(y_train.values[:,0])
            dataset['test_label'] = torch.from_numpy(y_test.values[:,0])

            model = KAN(width=[dataset['train_input'].shape[1],1,2], grid=6, k=3) # g=1, k=1

            def train_acc():
                return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

            def test_acc():
                return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
            #results = model.train(dataset, opt="LBFGS", steps=50, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(),lamb=00.9)
            results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
    if m!='KAN_1':
        # filter model predictions
        preds_tr=pd.DataFrame(preds_tr).transpose()
        preds_te=pd.DataFrame(preds_te).transpose()
        #print(preds_te)
        if fil==True:
            filt_chi=SelectKBest(chi2, k=k2).fit(preds_tr, y_train) #k=4
            preds_tr=filt_chi.transform(preds_tr)
            preds_te=filt_chi.transform(preds_te)
            names = filt_chi.get_support(indices=True)
            print(names)
            Filt_2.append(names)
            preds_tr=pd.DataFrame(preds_tr)
            preds_te=pd.DataFrame(preds_te)

        # gather predictions for training and testing
        # train and test the model
    if m=='KAN':

        
        dataset_p = {}
        dataset_p['train_input'] = torch.from_numpy(preds_tr.values)#pd.DataFrame(preds_tr).transpose().values)

        dataset_p['test_input'] = torch.from_numpy(preds_te.values)#pd.DataFrame(preds_te).transpose().values)
        dataset_p['train_label'] = torch.from_numpy(y_train.values[:,0])
        dataset_p['test_label'] = torch.from_numpy(y_test.values[:,0])


        model_p = KAN(width=[dataset_p['train_input'].shape[1],1,2], grid=6, k=3) # g=1, k=1

        def train_acc():
            return torch.mean((torch.argmax(model_p(dataset_p['train_input']), dim=1) == dataset_p['train_label']).float())

        def test_acc():
            return torch.mean((torch.argmax(model_p(dataset_p['test_input']), dim=1) == dataset_p['test_label']).float())
        #results = model.train(dataset, opt="LBFGS", steps=50, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(),lamb=00.9)
        results = model_p.train(dataset_p, opt="LBFGS", steps=100, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(),lr=0.1,lamb=0.009)
    if m=='SVM':
        dataset_p=preds_te
        model_p=svm.SVC()
        model_p=model_p.fit(preds_tr,y_train)
        
    if m=='MLP':
        dataset_p=preds_te
        model_p=MLPClassifier(hidden_layer_sizes=(1,2))
        model_p=model_p.fit(preds_te,y_test)
    if m=='KAN_1':
        models=[]
        model_p=model
        dataset_p=dataset

    return models,model_p,dataset,dataset_p,Filt_1,Filt_2,sampled_arrays


# CONFOUNDER CORRECTIONS

def CV_dataset_builder(X_train, y_train,cv,pipeline_biocov_reg):
    import statsmodels.api as sm
    # Creates the CV dataset according to the defined pipeline 
    # INPUT: predictors and target vectors (X_train,y_train), cv indexes of the original dataset, correction pipeline to fit
    # OUTPUT: predictor and target vectors in the CV_dataset shape, cv indexes in the CV dataset shape that points to the corrected folds
    # CV script :this part of the script builds the CV dataset according to pieplie 2 (models the effect of the target correlation on the correction)
 


    #cv = KFold(n_splits=n_splits)#you have to set manually n_repeats=1


    


    #X_train=pd.concat([X_train,y_train], axis=1)

    X_tr_tot=pd.DataFrame([])# total set that comprizes all the Kfold that we want to fit
    X_t_tot=pd.DataFrame([]) 
    y_tr_tot=pd.DataFrame([])
    y_t_tot=pd.DataFrame([])
    cv_gs=[]
    a=0
    b=0
    binary_var=['smoker','occasional_smoker','non_smoker','DRINKER'] # for mediator is correct because binary variables are not used
    #confounder_var=['SEX','AGE','Dx','batch']

    confounder_var=['SEX','AGE','ROP','ROD','batch']
    # since the indexes of the cv given by methods already done don't allow to create the correction inside the GS loop.
    # there is the need to build a dataset for the training and test split that contains all the folds. As well as an index array 
    # that correctly point to that dataset indexes

    ## APPLICATION OF THE CORRECTION FOR EVERY FOLD ##

    for train, test in cv.split(X_train,y_train): # for loop ove every k-fold subset 

        # Create subsets of data using K-fold cross validation for each iteration   
        X_tr, X_t= X_train.iloc[train].copy(), X_train.iloc[test].copy()
        y_tr, y_t = y_train.iloc[train].copy(), y_train.iloc[test].copy()
              
        #Divide data & covariates
        #data --> all the dataset apart from the covariates
        #covariates --> age,gender

        # SPLIT BINARY AND CONTINUOUS VARIABLES

        X_b_tr=X_tr[binary_var + confounder_var].copy()
        X_b_t=X_t[binary_var + confounder_var].copy()

        # drop binary variables from continuous dataset
        X_tr=X_tr.drop(binary_var,axis=1)
        X_t=X_t.drop(binary_var,axis=1)
    
        # Save confounders in other db & Remove them from the main data, to avoid overflow error
    
        # Save variables
        X_c_tr=X_tr[confounder_var].copy()
        X_c_t=X_t[confounder_var].copy()

        # drop them from the main dataset
        X_tr=X_tr.drop(confounder_var,axis=1).copy()
        X_t=X_t.drop(confounder_var,axis=1).copy()

  
        # CONTINUOUS AND BINARY DICTIONARY
        X_train_dict={'data': X_tr, 'covariates': X_c_tr}
        X_test_dict={'data': X_t, 'covariates': X_c_t}

        X_b_train_dict={'data': X_b_tr, 'covariates': X_c_tr}
        X_b_test_dict={'data': X_b_t, 'covariates': X_c_t}


        ###### CONTINUOUS ######
        # Now we apply the pipeline of the bicovariance to the predictors
        pipeline_biocov_reg.fit(X_train_dict) # Fit pipeline
        X_train_corr=pipeline_biocov_reg.transform(X_train_dict)# Apply fitted pipeline on train
  

        #We use the betas estimated in test and apply the correction on the validation 
        pipeline_biocov_reg.fit(X_train_dict) #See wether the result improoves if we don't use this if it does it's wrong
        X_test_corr=pipeline_biocov_reg.transform(X_test_dict) # Apply fitted pipeline on test  
 

        for i in range(0,len(X_train_corr)):
            X_tr.iloc[i]=X_train_corr[i].copy() #apply correction on train sample to the train set
        for i in range(0,len(X_test_corr)):
            X_t.iloc[i]=X_test_corr[i].copy() #apply correction on test sample to the test set

        ####### BINARY ########
        # Now we apply the pipeline of the bicovariance to the predictors

        # The pipeline in this case is only the logit regression

        # Since the logit model needs to be trained for each feature independently there is the
        # need of a for loop over the binary features
    
        binary_var=['smoker','occasional_smoker','non_smoker','DRINKER']
    
        #for feat in binary_var:
       
            #--This part is common for both train and test beacuse is the training of the regression--

            #est = sm.Logit(X_b_tr[feat],X_b_tr[confounder_var]) 
            #estimates=est.fit()
            #estimates.params[2]=0 # remove effect of Dx

            #---
        
            #for i in range(0,len(X_b_tr[feat])):
                # Substitue the logit value to the old one for train
            #    X_b_tr[feat].iloc[i]=max(0,X_b_tr[feat].iloc[i]-estimates.predict(X_b_tr[confounder_var].iloc[i])[0])
            #for i in range(0,len(X_b_t[feat])):
                # do the same for test
            #    X_b_t[feat].iloc[i]=max(0,X_b_t[feat].iloc[i]-estimates.predict(X_b_t[confounder_var].iloc[i])[0])
        
    
        # drop the cconfounders
        X_b_tr=X_b_tr.drop(confounder_var,axis=1).copy()
        X_b_t=X_b_t.drop(confounder_var,axis=1).copy()
    
        #### CONTINUOUS AND BINARY CONCAT ####
        X_tr=pd.concat([X_b_tr,X_tr],axis=1)
        X_t=pd.concat([X_b_t,X_t],axis=1)

    
        # update total train set
        X_tr_tot=pd.concat([X_tr_tot, X_tr], axis=0)
        y_tr_tot=pd.concat([y_tr_tot, y_tr], axis=0)
        #print('X_tr_tot.shape[0] =',X_tr_tot.shape[0]) --> debuger to see if the dataset is increasing in size
    
        #update the total validation set
        X_t_tot=pd.concat([X_t_tot, X_t], axis=0)
        y_t_tot=pd.concat([y_t_tot, y_t], axis=0)
    
        # update train_indexes & validation_indexes
        test_tot=np.linspace(a,a+len(X_tr)-1,dtype = int, num=len(X_tr))
        train_tot=np.linspace(b,b+len(X_t)-1,dtype = int, num=len(X_t))
        a=test_tot[-1]+1
        b=train_tot[-1]+1

        #update fold and cv
        fold = (test_tot,train_tot)
        cv_gs.append(fold)
    
    #print('X_tr_tot.shape[0] =',X_tr_tot.shape[0])  --> debugger to see if the dataset is increasing in size 

    # We will place all the validation set ad the end of the CV dataset after all the trsin folds, so we want to correct the validation indexes.
    # To do this we add the length of the total training dataset 
         
    for i in range(len(cv_gs)):
        for j in range(len(cv_gs[i][1])):
            cv_gs[i][1][j]+=len(X_tr_tot)

    #print('X_tr_tot.shape[0] =',X_tr_tot.shape[0]) --> debugger to see if the dataset is increasing in size
    X_train_gs=pd.concat([X_tr_tot,X_t_tot], axis=0)
    y_train_gs=pd.concat([y_tr_tot,y_t_tot], axis=0)
    #print('X_tr_tot.shape[0] =',X_tr_tot.shape[0]) --> debugger to see if the dataset is increasing in size 
    return X_train_gs,y_train_gs,cv_gs
  
