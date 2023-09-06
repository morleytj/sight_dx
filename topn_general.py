import sys
import pandas as pd

#columns in group df: numcases, did
#columns in patient df: dID, cc_status, ranking
#topn takes n as well as the did_perf (per did dataframe), and the per person
#for each distinct dID, find how often cases with that disease have it within the top n ranking
def topn(n, group_df, patient_df, id_col, case_count_col, group_col, target_col, rank_col):
    topn_dict = dict()
    unique_groups=group_df.loc[group_df[case_count_col]>0,group_col].unique()
    for d in unique_groups:
        #check if dID is same, cc status is positive, and ranking is <= n
        #get number of distinct cases in that situation
        num_correct=patient_df.loc[(patient_df[group_col]==d)&(patient_df[target_col]==1)&(patient_df[rank_col]<=n),id_col].nunique()
        total = patient_df.loc[(patient_df[group_col]==d)&(patient_df[target_col]==1),id_col].nunique()
        #calc statistic, assign to dict
        topn_stat=num_correct/total
        topn_dict[d]=topn_stat
    #create dataframe from dict, return it
    results_df = pd.DataFrame()
    results_df[group_col]=topn_dict.keys()
    results_df['top'+str(n)]=topn_dict.values()
    return results_df

#call format: python scriptname df_group df_patient n out id casec groupc targetc rankc
if __name__=='__main__':
    #df with group
    df_group = pd.read_csv(sys.argv[1],sep='|')
    #df with patient level data
    df_patient = pd.read_csv(sys.argv[2])
    n = int(sys.argv[3])
    outpath = str(sys.argv[4])
    id_col = str(sys.argv[5])
    case_count_col = str(sys.argv[6])
    group_col = str(sys.argv[7])
    target_col = str(sys.argv[8])
    rank_col = str(sys.argv[9])
    topn(n,df_group,df_patient,id_col,case_count_col,group_col,target_col,rank_col).to_csv(outpath,index=False)
