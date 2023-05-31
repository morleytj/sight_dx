import sys
import pandas as pd
import numpy as np
import scipy.stats as st
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
from joblib import dump, load
import pickle


#need to decide if matching or not
#if not matching in any way, then we can simply do random selection of N individuals from entire RD

'''
this function assumes that dID_colnum is the same between the two dataframes -- would need a dummy dID col in the complete controls
two stages
stage 1: duplicate cases as controls 
stage 2: select additional controls from the entire RD
'''
def select_controls(cases, dID_colnum, potential_controls, n_genet_control, n_dID_per_control, n_complete_control):
    #step 0: add case control marker (important as need to know the true status before we start duplicating)
    cases['cc_status'] = 1
    potential_controls['cc_status'] = 0
    #step 1: duplicate cases as controls
    #step 1a: select all the cases to be used
    sample_cases = cases.sample(n_genet_control)
    duplicated = []
    for i in range(n_genet_control):
        #step 1b: Select n_dID_per_control number of random unique dIDs that are not on selected case
        diseases = np.random.choice(cases.loc[cases.dID!=sample_cases.iloc[i]['dID'], 'dID'],n_dID_per_control)
        #step 1c: Create new rows for each dID selected, using the current case selected -- create a list of lists in "dupllicated"
        for disease in diseases:
            temp = list(sample_cases.iloc[i])
            temp[dID_colnum] = disease
            #change to control status
            temp[len(temp)-1] = 0 
            duplicated.append(temp)
    #step 2: select complete controls
    sample_controls = potential_controls.sample(n_complete_control)
    for j in range(n_complete_control):
       temp = list(sample_controls.iloc[j])
       #select random disease
       temp[dID_colnum] = np.random.choice(cases['dID'], 1)[0]
       duplicated.append(temp)
    #step 3: merge cases and controls
    controls_final = pd.DataFrame(duplicated, columns=cases.columns)#this was bad because there were columns in cases that weren't in the controls
    case_control = pd.concat([cases, controls_final])
    return case_control


def wide_df(phedf):
    wide = phedf.groupby(['PERSON_ID','PHECODE']).size().unstack().fillna(0).astype(int).reset_index()
    return wide

'''
input:
    long_phecode_df: longform df with phecodes for everyone, cases and controls
    cases: file containing person_id of cases and the year to censor by
        -has column "minyear" for censoring
function:
    remove codes of people in the cases file from the long_phecode_df where 
    the year of code entry is >= year of censoring. then make the whole thing wide? (this might be tough with memory reqs)
returns:
    wide_df_cen: censored version of the wide phecode dataframe
'''
def censor_codes(long_phecode_df, cases):
    #First extract the year for the phecodes
    long_phecode_df['ENTRY_YEAR']=long_phecode_df['ENTRY_DATETIME'].str[:4].astype(int)
    #Next merge on the minyear column for those who have it
    long_phecode_df = long_phecode_df.merge(cases[['PERSON_ID','minyear']], on='PERSON_ID', how='left')
    #now censor based on minyear
    long_phecode_df = long_phecode_df.loc[(long_phecode_df.minyear>long_phecode_df.ENTRY_YEAR)|(long_phecode_df.minyear.isna())].copy()
    #make it wide
    return wide_df(long_phecode_df[['PERSON_ID','ENTRY_DATETIME','PHECODE']])


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred):
     return confusion_matrix(y_true, y_pred)[1, 1]

def ppv(y_true, y_pred):
    if (fp(y_true, y_pred) + tp(y_true, y_pred))>0:
        return tp(y_true, y_pred)/(fp(y_true, y_pred) + tp(y_true, y_pred))
    else:
        return 0



'''
Input:
    need input of prematched df with omim columns
Output:
    full predictions
'''
def pred_pipeline(df, cpu_num, target_col_name,test_cc):
    #split out train test
    ######going to have to change this now, since the test will be split out prior to the pipeline
    #It has already been presplit
    #x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=[target_col_name]), df[target_col_name], test_size=.2, shuffle=True, stratify=df[target_col_name])
    x_train = df.drop(columns=target_col_name)
    y_train = df[[target_col_name]].values.ravel()
    #set pipeline
    pipe = Pipeline(steps=[('reduce_dim', None), ('classify', None)])
    reduce_dim_pca = [PCA(.95), None]
    #set search space
    param_grid = [
                {
                    'reduce_dim':reduce_dim_pca,
                    'classify':[LogisticRegression(solver='saga')],
                    'classify__C': st.loguniform(1e-5,100),
                    'classify__penalty': [None, 'l1', 'l2']
                },
                {
                    'reduce_dim':reduce_dim_pca,
                    'classify': [RandomForestClassifier()],
                    'classify__n_estimators': sp_randint(200,1100),
                    'classify__max_depth':sp_randint(3,250),
                    'classify__min_samples_split':sp_randint(2,10),
                    'classify__min_samples_leaf':sp_randint(1,10),
                    'classify__max_features': ['auto','sqrt','log2',None]
                },
                {
                    'reduce_dim':reduce_dim_pca,
                    'classify':[AdaBoostClassifier()],
                    'classify__n_estimators': sp_randint(200,1100),
                    'classify__learning_rate':st.loguniform(1e-5,1)
                },
                {
                    'classify':[MultinomialNB()]
                },
                {
                    'reduce_dim':reduce_dim_pca,
                    'classify':[GradientBoostingClassifier()],
                    'classify__learning_rate': st.reciprocal(1e-3, 5e-1),
                    'classify__max_depth': sp_randint(3,250),
                    'classify__max_leaf_nodes':sp_randint(10,200),
                    'classify__min_samples_leaf': sp_randint(1,200),
                    'classify__min_samples_split': sp_randint(2,200),
                    'classify__n_estimators': sp_randint(200,1100),
                    'classify__subsample': st.uniform(0.5,0.5)
                }
            ]
    #run pipeline
    splits = StratifiedKFold(n_splits=4, shuffle=True)
    score_custom = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn), 'precision_micro': 'precision_micro', 'f1': 'f1', 'auc': 'roc_auc', 'neg_brier_score': 'neg_brier_score', 'neg_log_loss': 'neg_log_loss', 'ppv': make_scorer(ppv), 'average_precision': 'average_precision'}
    search = RandomizedSearchCV(pipe, cv=splits, scoring=score_custom, refit='f1', param_distributions=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False, n_iter=2500)
    print(search)
    print(pipe)
    search.fit(x_train.drop(['PERSON_ID','dID'],axis=1), y_train)
    final_results_df = pd.DataFrame(search.cv_results_)
    best_est = search.best_estimator_
    print(best_est)
    print('\n')
    print(search.best_params_)
    print('\n')
    print(search.best_score_)
    print('\n')
    pipe.set_params(**search.best_params_)
    pipe.fit(x_train.drop(['PERSON_ID','dID'],axis=1), y_train)
    x_test = test_cc.drop(columns=target_col_name)
    y_test = test_cc[target_col_name].values.ravel()
    probs = pipe.predict_proba(x_test.drop(['PERSON_ID','dID'],axis=1))
    #preds = pipe.predict(x_test.drop(['PERSON_ID','dID'],axis=1))
    #test_ret_df = pd.DataFrame() #this is being added on wrong, index from x_test is not 0,1,2,3 and is matching on wrong, so that instead of going onto the preds and the probs in order its only adding onto a certain subset of indices, i think
    #test_ret_df['cc_status'] = y_test
    #test_ret_df['PERSON_ID']=x_test['PERSON_ID']
    #test_ret_df['dID']=x_test['dID']
    #test_ret_df['preds'] = preds
    #test_ret_df['probs'] = probs[:,1]
    #print(classification_report(y_test, preds))
    #save feature importances
    #fimps = pd.DataFrame()
    #fimps['importance']=pipe['classify'].feature_importances_
    #fimps['feature_name']=list(x_train.drop(['PERSON_ID','dID'],axis=1).columns)
    #return results
    return final_results_df, pipe, probs#, test_ret_df#, fimps


def add_omim_col(df, omimdf):
    #get dataframe with added DID for matching (after case control matching), and df hat has omim code columns filled in for all DID
    return df.merge(omimdf, on='dID', how='inner')



###need to read in long instead of wide dataframe (nogenet) and censor out the rows that don't meet the criteria
###criteria: either (not matching the personid of a case) or (is matching personid of a case and date of entry is <minyear)
'''
input: gendf dbdf omim_df dID_col n_genet_control n_dID_per_control n_complete_control cpus target output_path
    gendf: dataframe containing all genet individuals
        The cases are all individuals with genetdx or clindx positive
        Controls are selected on a sliding scale as a hyperparameter (from where?)
    dbdf: wide dataframe containing phecodes for all individuals, from RD. To be used in selecting cases.
    omim_df: wide dataframe containing mapping between dID and omim phecodes
output:
'''
if __name__=="__main__":
    gendf = pd.read_csv(sys.argv[1])#,encoding='ISO-8859-1')
    ##remove dID=0
    ##gendf = gendf.loc[gendf.dID!=0]
    dbdf = pd.read_csv(sys.argv[2],dtype={'PHECODE':str})
    omim_df = pd.read_csv(sys.argv[3], sep='|')
    #remove gendf rows where dID isn't in omim_df
    print(gendf.shape)
    gendf = gendf.loc[gendf.dID.isin(omim_df.dID)]
    print(gendf.shape)
    #Censor and widen dataframe
    dbdf = censor_codes(dbdf, gendf)
    #get col headers for later
    col_headers = list(dbdf.columns)
    #get omim cols
    col_headers = col_headers + list(omim_df.columns) + ['cc_status']
    dID_col = int(sys.argv[4])
    n_genet_control = int(sys.argv[5])
    n_dID_per_control = int(sys.argv[6])
    n_complete_control = int(sys.argv[7])
    cpus = int(sys.argv[8])
    target = sys.argv[9]
    output_path = sys.argv[10]
    test_df = pd.read_csv(sys.argv[11])#test set version of gendf
    test_phe = pd.read_csv(sys.argv[12],dtype={'PHECODE':str})#test cases and controls
    #censor and widen test_phe
    test_phe = censor_codes(test_phe, test_df)
    print('Loaded inputs', flush=True)
    #exclude test set from cases and potential controls
    dbdf = dbdf.loc[~dbdf.PERSON_ID.isin(test_phe.PERSON_ID)].copy()
    cases = gendf.merge(dbdf,on='PERSON_ID', how='inner') #changed from left to inner
    test_df['cc_status']=1
    test_cc = test_df.merge(test_phe,on='PERSON_ID',how='outer')
    #get na indices
    control_test_indices=test_cc.loc[test_cc['dID'].isna()].index
    #for each na index, select a random dID from the training set (so ones we have seen before in training)
    control_test_dids = []
    for x in control_test_indices:
        control_test_dids.append(np.random.choice(cases['dID'], 1)[0])
    print(test_cc.loc[test_cc.cc_status==1,'dID'],flush=True)
    test_cc.loc[test_cc['dID'].isna(),'dID']=control_test_dids#should fill in NA values for dID from the controls as a randomly chosen dID from case set
    print(test_cc.loc[test_cc.cc_status==1,'dID'],flush=True)#print out the did of CC_STATUS=1 before and after changing controls, should not change the dID, so should be same twice
    test_cc['cc_status']=test_cc['cc_status'].fillna(0)
    #fix so that cases and controls have same columns
    #get rid of columns that aren't going to be in the final output
    cases=cases.drop(columns=[col for col in cases.columns if col not in col_headers])
    test_cc=test_cc.drop(columns=[col for col in cases.columns if col not in col_headers])
    controls = dbdf.loc[~dbdf.PERSON_ID.isin(gendf.PERSON_ID)].copy()
    controls.insert(loc=dID_col,column='dID',value=0)
    #match case control
    #make sure case and control have same columns in same order, throw error otherwise
    if list(cases.columns)!=list(controls.columns):
        raise Exception('Case and control columns are not equal!')
    ccdf = select_controls(cases, dID_col, controls, n_genet_control, n_dID_per_control, n_complete_control)
    print('Matched controls', flush=True)
    #add omim col
    omim_add = add_omim_col(ccdf, omim_df)
    #omim_add = omim_add.fillna(0)
    print('Added omim cols', flush=True)
    #before passing into pipeline, need to drop unneccessary columns
    #need to keep omim columns, cc_status, phecode columns, dID and person_id
    omim_add.to_csv(output_path+'omim_add.csv',index=False)
    omim_add = omim_add[col_headers]
    test_omim_add = add_omim_col(test_cc,omim_df)
    for col in col_headers:
        if col not in test_omim_add:
            test_omim_add[col]=0
    test_omim_add = test_omim_add[col_headers].fillna(0)
    test_omim_add.to_csv(output_path+'test_omim_add.csv',index=False)
    #Need to write col_headers to file
    with open(output_path+'col_headers.pkl', 'wb') as f:
        pickle.dump(col_headers, f)
    #run pipeline
    final_res, pipeline, probs = pred_pipeline(omim_add, cpus, target, test_omim_add)
    print('Finished pipeline', flush=True)
    output_res = output_path+"final_results.csv"
    #output_probs = output_path+"final_test_set_df.csv"
    output_model = output_path+"full_pipeline.joblib"
    #output_imps = output_path+"feature_importances.csv"
    final_res.to_csv(output_res,index=False)
    #test_ret.to_csv(output_probs, index=False)
    dump(pipeline, output_model)
    np.savetxt(output_path+"test_probs.txt",probs)
    #feature_importance.to_csv(output_imps, index=False)
