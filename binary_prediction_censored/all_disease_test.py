import pandas as pd
import pickle
import sys
import joblib
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


#omimadd cols model omim out
if __name__=="__main__":
    #load in data
    omimadd = pd.read_csv(sys.argv[1])
    cols = []
    with open(sys.argv[2], 'rb') as f:
        cols = pickle.load(f)
    for c in omimadd.columns:
        if c in cols and "_omim" not in c and "dID" not in c and "cc_status" not in c and "PERSON_ID" not in c:
            omimadd[c]=omimadd[c].astype("int8")
    model = joblib.load(sys.argv[3])
    omim = pd.read_csv(sys.argv[4], sep="|")
    for omim_feat in omim.columns:
        if "_omim" in omim_feat:
            omim[omim_feat]=omim[omim_feat].astype("int8")
    outpath = sys.argv[5]
    feat_cols = [x for x in cols if "_omim" not in x and "dID" not in x and "cc_status" not in x]
    print(feat_cols)
    print(omimadd.shape)
    cross = omimadd[feat_cols].merge(omim,how='cross')
    print(cross.shape)
    #predict probabilities
    cols_t = [x for x in cols if "cc_status" not in x]
    probs = model.predict_proba(cross[cols_t].drop(['PERSON_ID','dID'],axis=1))
    cross['case_prob'] = probs[:,1]
    cross.to_csv(outpath+'all_disease_testset_res_df_may11.csv',index=False) 
