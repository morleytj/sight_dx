#topn takes n as well as the did_perf (per did dataframe), and the per person
#for each distinct dID, find how often cases with that disease have it within the top n ranking
def topn(n, dids, pids):
    topn_dict = dict()
    unique_dids=dids.loc[dids.numcases>0,'did'].unique()
    for d in unique_dids:
        #check if dID is same, cc status is positive, and ranking is <= n
        #get number of distinct cases in that situation
        num_correct=pids.loc[(pids.dID==d)&(pids.cc_status==1)&(pids.ranking<=n),'PERSON_ID'].nunique()
        total = pids.loc[(pids.dID==d)&(pids.cc_status==1),'PERSON_ID'].nunique()
        #calc statistic, assign to dict
        topn_stat=num_correct/total
        topn_dict[d]=topn_stat
    #create dataframe from dict, return it
    results_df = pd.DataFrame()
    results_df['did']=topn_dict.keys()
    results_df['top'+str(n)]=topn_dict.values()
    return results_df
