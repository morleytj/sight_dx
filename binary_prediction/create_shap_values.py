import sys
import pandas as pd
import numpy as np
import scipy.stats as st
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, roc_auc_score, average_precision_score
from scipy.stats import randint as sp_randint
import shap


#python scriptname omim features dd omadd model out trainset
if __name__=="__main__":
    omim = pd.read_csv(sys.argv[1],sep='|')
    for omim_feat in omim.columns:
        if "_omim" in omim_feat:
            omim[omim_feat]=omim[omim_feat].astype("int8")
    features = pd.read_csv(sys.argv[2],dtype=str)
    cols = ['PERSON_ID']+list(features['feature'])
    cols = [x for x in cols if "_omim" not in x]
    print('loaded cols',flush=True)
    coldict = dict()
    for c in cols:
        if c=="PERSON_ID":
            coldict[c]=str
        else:
            coldict[c]="int8"
    dd = pd.read_csv(sys.argv[3],sep=',', usecols=cols, dtype=coldict)
    did_id = pd.read_csv(sys.argv[4],sep=',',usecols=['PERSON_ID','dID'])
    print(dd.head())
    print(dd.columns)
    print('loaded dd',flush=True)
    ##need to select only AADC for now
    omimaadc = omim.loc[omim.dID==608643]
    ##next exclude everyone who is in training set
    #load in dataframe with column 'PERSON_ID', exclude all person_id in that dataframe
    trainset = pd.read_csv(sys.argv[7],index_col=0,dtype=str)
    df = dd.loc[~dd.PERSON_ID.isin(trainset.PERSON_ID)]
    #now do merge
    df = df.merge(omimaadc,how='cross')
    print('merged cross',flush=True)
    print(df.head())
    #need a subsample of individuals from the training set
    subsample=dd.loc[dd.PERSON_ID.isin(trainset.PERSON_ID)].sample(n=500)
    subsample = subsample.merge(omim,how='cross')
    print('sample training done', flush=True)
    model = load(sys.argv[5])['classify']
    #make explainer
    explainer = shap.Explainer(model, subsample[list(features['feature'])])
    print('explainer made', flush=True)
    #generate shap values for selected individuals
    shap_values = explainer.shap_values(df[list(features['feature'])])
    print('values calced', flush=True)
    ####get values, cast into pandas df, save them?
    shap_one = pd.DataFrame(shap_values, columns=list(features['feature']))
    shap_one['PERSON_ID']=df['PERSON_ID']
    shap_one.to_csv(sys.argv[6], index=False)
    print(explainer.expected_value)
    #don't need probs anymore
    #probs = model.predict_proba(dd[list(features['feature'])])
    #print('did prediction',flush=True)
    #res = dd[['PERSON_ID','dID']].copy()
    #res['case_prob'] = probs[:,1]
    #res['pid_did']=res['PERSON_ID'].astype(str)+res['dID'].astype(str)
    #did_id['pid_did'] = did_id['PERSON_ID'].astype(str)+did_id['dID'].astype(str)
    #res['in_orig_set']=0
    #res.loc[res.pid_did.isin(did_id.pid_did),'in_orig_set']=1
    #print('saving',flush=True)
    #res.to_csv(sys.argv[6],index=False,sep='|')
