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
    ##the save function of tree explainers is broken
    #expout = open(out+"TreeExplainer_File.out",'wb')
    #explainer.save(expout)
    #expout.close()
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
    sdf: contains shap values
    pid: Person_id of the person we want plotted
    did: dID we want plotted for the person we want plotted
    expected_val: expected value from the explainer
    cols: columns in order for the classifier
    out: path to save results to
    contr: contribution threshold as in shap force plots, set default to 0.05
'''
def gen_force_plot(df, sdf, pid, did, expected_val, cols, out, contr=0.05):
    cols = [c for c in cols if c not in ['cc_status','dID','case_prob','PERSON_ID']]
    feats = df.loc[(df.person_id==int(pid))&(df.dID==int(did)),cols].values
    row = sdf.loc[(df.person_id==int(pid))&(df.dID==int(did)),cols].values
    shap.plots.force(expected_val, shap_values=row, features=feats, matplotlib=True,show=False, feature_names=cols, contribution_threshold=contr)
    plt.savefig(out+str(person_id)+'_'+str(did)+'_force_plot.png')


# all_disease_5078940.csv
'''
This function takes in PID and DID, loads the appropriate filepath using the format above, and generates a foce plot from it.
To generate this it gets the shap value of this row, which means needing to train the explainer each time, but that doesn't take too long
Input
Output
'''
def force_plot_id(pid, did, cols, clf, background, out):#, contr=0.05):
    cc = [c for c in cols if c not in ['cc_status','dID','case_prob','PERSON_ID']]
    explainer = shap.explainers.Tree(clf, data=background[cc])
    df = pd.read_csv('../../results/person_id_files_all_disease/all_disease_'+str(pid)+'.csv')
    feats = df.loc[(df.dID==int(did)),cc].values
    shap_vals = explainer.shap_values(feats)[1]
    shap.plots.force(explainer.expected_value[1], shap_values=shap_vals, features=feats, matplotlib=True, show=False, feature_names=cc)#, contribtion_threshold=contr)
    plt.savefig(out+str(pid)+'_'+str(did)+'_force_plot.png')


#python make_shap_values.py ../../results/may9_binary_prediction_cen_col_headers.pkl ../../results/may9_binary_prediction_cen_full_pipeline.joblib ../../results/shapres/force_plot_ ../../results/may9_binary_prediction_cen_omim_add.csv pid did
#python make_shap_values.py columns clf out bg pid did (optional_contr)
if __name__=="__main__":
    fc = open(sys.argv[1], 'rb')
    col = pickle.load(fc)
    fc.close()
    clf = joblib.load(sys.argv[2])['classify']
    out = sys.argv[3]
    bg = pd.read_csv(sys.argv[4])
    personid=sys.argv[5]
    did = sys.argv[6]
    #if len(sys.argv)>7:
    #    contr = float(sys.argv[7])
    #    force_plot_id(personid, did, col, clf, bg, out, contr)
    #else:
    force_plot_id(personid, did, col, clf, bg, out)

'''
#python scriptname genshap cols df clf out bg
#or
#python scriptname forceplot cols df clf out personid did expval fnames shapdf (optional)contr
#load in and format files, run the runshap function
if __name__=="__main__":
    fc = open(sys.argv[2], 'rb')
    col = pickle.load(fc)
    fc.close()
    df = pd.read_csv(sys.argv[3])
    clf = joblib.load(sys.argv[4])['classify']
    out = sys.argv[5]
    if sys.argv[1]=='genshap':
        bg = pd.read_csv(sys.argv[6])
        sv, exp = runshap(df, col, clf, bg, out)
    elif sys.argv[1]=='forceplot_depr':
        personid=sys.argv[6]
        did=sys.argv[7]
        expval=int(sys.argv[8])
        #load in shap values
        sdf = pd.read_csv(sys.argv[9])
        if len(sys.argv)>10:
            contr = float(sys.argv[10])
            gen_force_plot(df, sdf, personid, did, expval, col, fnames, out, contr)
        gen_force_plot(df, sdf, personid, did, expval, col, out)
    elif sys.argv[1]=='forceplot':
        personid=sys.argv[6]
        did=sys.argv[7]
        if len(sys.argv)>8:
            contr = float(sys.argv[8])
            force_plot_id(personid, did, col, clf, bg, out,contr)
        else:
            force_plot_id(personid, did, col, clf, bg, out)
    else:
        print("Input command not recognized. Please use either 'genshap' or 'forceplot'.")
'''
