import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

####################### cross validation #########################

def cv(dirpath, save_path, dataset, nfold):
    years = [3,5]
    name = ['id','time','status','class']
    data = pd.read_csv(dirpath+f'{dataset}_info.csv',usecols=name[:-1]).values
    for year in years:
        surv  = []
        for i,temp in enumerate(data):
            if temp[2]==-1:
                surv.append(list(data[i])+[2])
                continue
            if temp[2]==1 and int(temp[1]) < year*365:
                surv.append(list(data[i])+[1])
            elif int(temp[1]) >= year*365:
                surv.append(list(data[i])+[0])
            else:
                surv.append(list(data[i])+[2])
        df = pd.DataFrame(columns=name,data=surv)
        save_path = save_path + f'{dataset}_{year}y'
        if not os.path.exists(save_path): os.makedirs(save_path)
        df.to_csv(save_path + '.csv',index=None)

        x = df.values
        y = df['class'].values

        kf = StratifiedKFold(n_splits=nfold,random_state=23,shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(x, y)):
            res_train = pd.DataFrame(data=x[train_index])
            res_test = pd.DataFrame(data=x[test_index])
            writer = pd.ExcelWriter(save_path+f'/data_{i}.xlsx')
            eval('res_train').to_excel(excel_writer=writer, sheet_name='train', index=False)
            eval('res_test').to_excel(excel_writer=writer, sheet_name='test', index=False)
            writer.save()
            writer.close()
    return

if __name__ == "__main__":
    dirpath = './data/clinical/'
    save_path = './splits/5_fold_cv/'
    data_list = ['BLCA','COAD','LIHC']
    for dataset in data_list:
        cv(dirpath, save_path, dataset, nfold=5)
