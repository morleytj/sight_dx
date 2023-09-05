import pandas as pd
import sys

if __name__=="__main__":
    main_df = pd.read_csv(sys.argv[1])
    out_path = sys.argv[2]
    unique_pid = main_df['PERSON_ID'].unique()
    for p in unique_pid:
        main_df.loc[main_df.PERSON_ID==p].to_csv(out_path+str(p)+'.csv',index=False)
