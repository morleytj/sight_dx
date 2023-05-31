import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import sys


'''
Input:
    cols: in order list of columns for accessing dataset for classifier
    clf: loaded and trained classifier (written under assumption it is tree based)
    df: actual data, contains all columns in cols along with dID, person_id, and may contain case_prob
    out: path to save output to
Output:
    shapvals: dataframe with shap values for each input row and column
    explainer: explainer from shap
    also saves file of the explainer to out, as well as the shap values
'''
def runshap(df, cols, clf, out):
    explainer = shap.explainers.Tree(clf)
    expout = open(out+"TreeExplainer_File.out")
    explainer.save(expout)
    expout.close()
    shap_vals = explainer(df[[c for c in cols if c not in ['cc_status','dID','case_prob','PERSON_ID']]])
    print('Explainer expected value:')
    print(explainer.expected_value)
    return shap_vals, explainer

'''
Input:
    df: contains actual data for some number of individuals, at least including the person_id/dID combo of interest
    person_id: Person_id of the person we want plotted
    did: dID we want plotted for the person we want plotted
    expl: Explainer object, fully loaded
    cols: columns in order for the classifier
    out: path to save results to
'''
def genplot(df, person_id, did, expl, cols, out):
    #


#load in and format files, run the runshap function
if __name__=="__main__":
    #fc = open(colsloc, 'rb')
