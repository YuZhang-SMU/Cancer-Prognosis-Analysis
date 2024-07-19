import pandas as pd
import argparse
import utils.c1_core_utils as c1
import utils.c2_core_utils as c2
import time
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--gpu', type=str, default="2", help='GPU')
parser.add_argument('--multi_gpu', type=int, default=0, help='if multiple GPU')
parser.add_argument('--dataset', type=str, default="LIHC", help='data')
parser.add_argument('--fold', type=int, default=0, help='cross validation')

####################### Curriculum 1 #########################
parser.add_argument('--training', type=int, default=1, help='if training')
parser.add_argument('--encoding', type=int, default=1, help='if encoding instance')
parser.add_argument('--branch_epoch', type=list, default=[50,50,50], help='The epoch of each branch')
parser.add_argument('--iter', type=int)
parser.add_argument('--iters', type=list, default=[32,16,16])
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--sinst', type=int, default=20, help='The size of instance')
parser.add_argument('--sbag', type=int, default=-1, help='The size of max bag')
parser.add_argument('--lr', default=[1e-5,3e-6,3e-6], type=float, help='learning rate')
parser.add_argument('--l2w', default=1e-5, type=float, help='weight penalty rate')
parser.add_argument('--beta_o', default=1e-5, type=float, help='The weight coefficient')
parser.add_argument('--wscale', default=[1,1,1], type=list)
parser.add_argument('--spath', default='./results/curriculum_1/', type=str)

####################### Curriculum 2 #########################
parser.add_argument('--training_c2', type=int, default=1, help='if training')
parser.add_argument('--nepochs_c2', type=int, default=1000, help='The maxium number of epochs to train')
parser.add_argument('--iters_c2', type=list, default=16)
parser.add_argument('--lr_c2', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--l2w_c2', default=1e-5, type=float, help='weight penalty rate')
parser.add_argument('--beta_s', default=1e-3, type=float, help='The weight coefficient')
parser.add_argument('--k', type=int, default=3, help='top k')
parser.add_argument('--spath_c2', default='./results/curriculum_2/', type=str)

args = parser.parse_args()

def main(args):
    for i in range(5):
        
        args.fold = i
        data_path = './splits/5_fold_cv/{}_3y/data_{}.xlsx'.format(args.dataset, args.fold)
        train_df = pd.read_excel(data_path, sheet_name='train', names=['id','time','status','class'])
        train_pid, val_pid, _, _ = train_test_split(train_df,train_df['class'],test_size=0.25,random_state=23)
        test_df = pd.read_excel(data_path, sheet_name='test', names=['id','time','status','class'])
        
        ####################### Curriculum 1 #########################
        if args.training:
            c1.train(train_pid, val_pid, args=args)
            
        if args.encoding:
            c1.inst_encoding(train_df, test_df, args=args)
            

        data_path = args.spath+'{}/fold_{}.xlsx'.format(args.dataset, args.fold)
        train_df = pd.read_excel(data_path, sheet_name='train')
        train_pid, val_pid = train_df[train_df[0].isin(train_pid['id'])], train_df[train_df[0].isin(val_pid['id'])]
        test_df = pd.read_excel(data_path, sheet_name='test')

        ####################### Curriculum 2 #########################
        if args.training_c2:
            c2.train(train_pid, val_pid, args)
        c2.evaluation(test_df, args)
                
    return

if __name__ == "__main__":
    while True:
        start = time.time()
        results = main(args)
        end = time.time()
        print("finished!")
        print('Spending Time: %f seconds' % (end - start))
