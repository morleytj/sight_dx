import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import sys
import matplotlib.pyplot as plt


'''
Input:
    cols: in order list of columns for accessing dataset for classifier
    clf: loaded and trained classifier (written under assumption it is tree based)
    df: actual data, contains all columns in cols along with dID, person_id, and may contain case_prob
    background: dataframe containing the background data for explainer generation
    out: path to save output to
Output:
    shapvals: dataframe with shap values for each input row and column
    explainer: explainer from shap
    also saves file of the explainer to out, as well as the shap values
'''
def runshap(df, cols, clf, background, out):
    cc = [c for c in cols if c not in ['cc_status','dID','case_prob','PERSON_ID']]
    explainer = shap.explainers.Tree(clf, data=background[cc])
    expout = open(out+"TreeExplainer_File.out")
    explainer.save(expout)
    expout.close()
    shap_vals = explainer.shap_values(df[cc])
    #put shap values into a dataframe
    #use [1] since we're basing it around probability of being case, rather than control
    shap_val_df = pd.DataFrame(shap_vals[1],columns=cc)
    #add columns for dID and PERSON_ID
    shap_val_df['PERSON_ID'] = df['PERSON_ID']
    shap_val_df['dID'] = df['dID']
    #save shap values
    shap_val_df.to_csv(out+'shap_value_dataframe.csv',index=False)
    print('Explainer expected value:')
    print(explainer.expected_value)
    return shap_val_df, explainer

'''
Input:
    df: contains actual data for some number of individuals, at least including the person_id/dID combo of interest
    person_id: Person_id of the person we want plotted
    did: dID we want plotted for the person we want plotted
    expected_val: expected value from the explainer
    cols: columns in order for the classifier
    fnames: feature names in clf order to be displayed on plot
    out: path to save results to
'''
def gen_force_plot(df, person_id, did, expected_val, cols, fnames, out):
    #
    shap.plots.force(expected_val, shap_values=row, features=df[cols], matplotlib=True,show=False, feature_names=fnames)
    plt.savefig(out+str(person_id)+'_'++str(did)+'_force_plot.png')


#python scriptname genshap --
#or
#python scriptname forceplot --
#load in and format files, run the runshap function
if __name__=="__main__":
    if sys.argv[1]='genshap':
        #
    elif sys.arg[1]='forceplot':
        #
    else:
        print("Input command not recognized. Please use either 'genshap' or 'forceplot'.")
    #fc = open(colsloc, 'rb')
