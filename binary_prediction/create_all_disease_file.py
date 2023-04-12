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


#python scriptname omim features dd omadd model out
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
    dd = dd.merge(omim,how='cross')
    print('merged cross',flush=True)
    print(dd.head())
    model = load(sys.argv[5])
    probs = model.predict_proba(dd[list(features['feature'])])
    print('did prediction',flush=True)
    res = dd[['PERSON_ID','dID']].copy()
    res['case_prob'] = probs[:,1]
    res['pid_did']=res['PERSON_ID'].astype(str)+res['dID'].astype(str)
    did_id['pid_did'] = did_id['PERSON_ID'].astype(str)+did_id['dID'].astype(str)
    res['in_orig_set']=0
    res.loc[res.pid_did.isin(did_id.pid_did),'in_orig_set']=1
    print('saving',flush=True)
    res.to_csv(sys.argv[6],index=False,sep='|')

def predall():
    omim = pd.read_csv(sys.argv[1],sep='|')
    for omim_feat in omim.columns:
        if "_omim" in omim_feat:
            omim[omim_feat]=omim[omim_feat].astype("int8")
    features = pd.read_csv(sys.argv[2],dtype=str)
    cols = ['PERSON_ID']+list(features['feature'])
    cols = [x for x in cols if "_omim" not in x]
    print('loaded cols',flush=True)
    #load in just the person id and the non omim cols from the omim add version
    coldict = dict()
    for c in cols:
        if c=="PERSON_ID":
            coldict[c]=str
        else:
            coldict[c]="int8"
    omadd = pd.read_csv(sys.argv[3],sep=',', usecols=cols,dtype=coldict)
    print(omadd.head())
    print(omadd.columns)
    print('loaded omadd',flush=True)
    #drop duplicates
    dd = omadd.drop_duplicates()
    #now join with cartesian cross product
    dd = dd.merge(omim,how='cross')
    print('merged cross',flush=True)
    print(dd.head())
    #now need to load in model and do predictions (use feature importance list for col order)
    model = load(sys.argv[4])
    probs = model.predict_proba(dd[[x for x in list(features['feature']) if x in dd.columns]])[:,1]
    print('did prediction',flush=True)
    res = dd[['PERSON_ID','dID']].copy()
    res['case_prob'] = probs
    res['pid_did']=res['PERSON_ID'].astype(str)+res['dID'].astype(str)
    omadd['pid_did'] = omadd['PERSON_ID'].astype(str)+omadd['dID'].astype(str)
    res['in_orig_set']=0
    res.loc[res.pid_did.isin(omadd.pid_did),'in_orig_set']=1
    print('saving',flush=True)
    res.to_csv(sys.argv[5],index=False,sep='|')


