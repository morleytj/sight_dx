f=open('performance_by_dID_cleaned.csv','w')
for did in cleaned.dID.unique():
    if len(cleaned.loc[cleaned.dID==did,'cc_status'].unique())>1:
        f.write(str(did)+','+str(roc_auc_score(cleaned.loc[cleaned.dID==did,'cc_status'], cleaned.loc[cleaned.dID==did,'censored_cases_uncen_controls_prob']))+'\n')
f.close()
