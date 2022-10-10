import pandas as pd
import os

def clinical_screen(dirpath, wsipath, dataset):
    frame = pd.read_csv(dirpath+f'{dataset}_clinical.tsv', sep='\t')
   
    #### Delete the same rows
    data = frame.drop_duplicates(subset=['case_submitter_id'], keep='first', inplace=False)
    data.reset_index(drop=True, inplace=True)
    
    #### Rename the column name
    columns_dict = {"case_submitter_id":"id",
                    "age_at_index":"age",
                    "gender":"gender",
                    "race":"race",
                    "vital_status":"vital_status",
                    "days_to_death":"days_to_death",
                    "days_to_last_follow_up":"days_to_last_follow_up",
                    "ajcc_pathologic_stage":"tumor_stage",
                    'ajcc_pathologic_t':'tumor_T'
    }
    
    df = pd.DataFrame(data, columns=columns_dict.keys())
    df.rename(columns=columns_dict, inplace=True) 
    
    #### Traverse all WSI files
    filter = ['.svs']
    wsiid = []
    for maindir, subdir, file_name_list in os.walk(wsipath):
        for filename in file_name_list:
            ext = os.path.splitext(filename)[1]
            if ext in filter:
                wsiid.append(filename[0:12])

    print(len(wsiid))
    
    #### Extract survival information
    status, time = [], []
    for ind,i in enumerate(df['vital_status']):
        if i == 'Alive' and df['days_to_last_follow_up'][ind]!="'--":
            status.append(0)
            time.append(df['days_to_last_follow_up'][ind])
        elif i == 'Dead' and df['days_to_death'][ind]!="'--":
            status.append(1)
            time.append(df['days_to_death'][ind])
        else:
            status.append(-1)
            time.append(df['days_to_last_follow_up'][ind])        
    df['time'] = time
    df['status'] = status

    #### Creen patients
    inds = []
    for ind,i in enumerate(df['id']):
        if i not in wsiid:
            inds.append(ind)
    df.drop(df.index[inds],inplace=True)

    #### Write the renamed data to a file
    filepath = dirpath+f'{dataset}_info.csv'
    df_columns = pd.DataFrame([list(df.columns)])
    df_columns.to_csv(filepath, mode='w', header=False, index=0) 
    df.to_csv(filepath, mode='a', header=False, index=0)
    
    return

if __name__ == "__main__":
    data_list = ['BLCA','COAD','LIHC']
    dirpath = './data/clinical/'
    wsipath = './data/svs/'
    for dataset in data_list:
        clinical_screen(dirpath, wsipath, dataset)
