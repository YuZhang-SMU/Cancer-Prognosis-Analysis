import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import random
import os
from utils.Early_Stopping import EarlyStopping
from utils.metrics import _neg_partial_log, cox_log_rank, calculate_cindex, roc
from models.model import SurvModel, CPCModel

####################### Curriculum 2 #########################

def train(train_df, val_df, args):
    device = torch.device(f'cuda:{args.gpu}')
    model = SurvModel(args.k).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_c2, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, verbose=True)    
    
    model_save_path = f'{args.spath_c2 + args.dataset}/models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_save_path += '/c2_fold_{}.pth'.format(args.fold)
    early_stopping = EarlyStopping(model_path=model_save_path, patience=100, verbose=True)
    
    train_list, val_list = train_df.values.tolist(), val_df.values.tolist()
    dir_pt = f'{args.spath+args.dataset}/inst_fold_{args.fold}/'
    train_data = []
    for data in train_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt')
        except:
            continue
        if fea.shape[0] < args.k: continue
        bag_list = [fea, data[1], data[2], data[-1]]
        train_data = train_data + [bag_list]

    val_data = []
    for data in val_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt')
        except:
            continue
        if fea.shape[0] < args.k: continue
        bag_list = [fea, data[1], data[2]]
        val_data = val_data + [bag_list]

    for epoch in range(args.nepochs_c2):
        train_epoch(epoch, args.iters_c2, model, optimizer, train_data, args)
        valid_loss, val_ci, val_p, val_auc = prediction(model, val_data, args)
        scheduler.step(valid_loss)
    
        early_stopping(val_ci, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    _, c_index, p_value, auc = prediction(model, val_data, args)

    return

def train_epoch(epoch, iters, model, optimizer, train_data, args, measure=1, verbose=1):
    device = torch.device(f'cuda:{args.gpu}')
    cpc_model = CPCModel().to(device)
    model.train()
    cpc_model.train()

    survtime_iter, status_iter, survtime_all, status_all = [], [], [], []
    zw, c, hazard = [], [], []

    iter = 0

    loss_nn_all = []
    random.shuffle(train_data)
    tbar = tqdm(train_data, desc='\r')

    for i_batch, data in enumerate(tbar):
        # loading data
        bag_tensor, time, status, h = data
        bag_tensor = bag_tensor.to(device)
        
        hazard.append(h)
        survtime_iter.append(time/30.0)
        status_iter.append(status)
        
        # hard_weight = 1 + bag_tensor[:,-1] * (epoch / args.nepochs_c2)
        # ===================forward=====================
        y_pred, zw_temp, c_temp = model(bag_tensor[:,:-2],training=True)
        zw.append(zw_temp)
        c.append(c_temp)

        if iter == 0:
            y_pred_iter = y_pred
        else:
            y_pred_iter = torch.cat([y_pred_iter, y_pred])
        iter += 1

        if iter % iters == 0 or i_batch == len(train_data)-1:
            if np.max(status_iter) == 0:
                print("encounter no death in a batch, skip")
                continue

            optimizer.zero_grad()
            # =================== Cox loss =====================
            loss_surv = _neg_partial_log(y_pred_iter, np.asarray(survtime_iter), np.asarray(status_iter), device)
            
            # =================== TCL loss =====================
            hazard = torch.FloatTensor(hazard).to(device)
            zw = torch.cat(zw)
            c = torch.cat(c)
            zw_0 = zw[hazard==0]
            zw_1 = zw[hazard==1]
            c = torch.cat([c[hazard==0],c[hazard==1]])
            
            nce_loss = cpc_model(zw_0=zw_0, zw_1=zw_1, c=c, device=device)

            # =================== CSA loss =====================
            l1_reg, l1_wa = None, None
            for param in model.state_dict():
                if param[:2] == 'Wa':
                    wp = model.state_dict()[param]
                    if l1_wa is None:
                        l1_wa = torch.abs(wp).sum()
                    else:
                        l1_wa = l1_wa + torch.abs(wp).sum()
                        
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()            
            
            # =================== total loss =====================   
            loss = loss_surv + nce_loss + args.beta_s * l1_wa + args.l2w_c2 * l1_reg
            
    # ===================backward====================
            loss.backward()
            optimizer.step()

            y_pred_iter = None
            survtime_iter, status_iter = [], []
            loss_nn_all.append(loss.data.item())
            iter = 0
            zw, c, hazard = [], [], []

        # ===================measure=====================
        if i_batch == 0:
            y_pred_all = y_pred[:,-1].detach().cpu()
        else:
            y_pred_all = torch.cat([y_pred_all, y_pred[:,-1].detach().cpu()])
        survtime_all.append(time)
        status_all.append(status)

    if measure:
        pvalue_pred = cox_log_rank(y_pred_all, np.asarray(status_all), np.asarray(survtime_all))
        c_index = calculate_cindex(y_pred_all, np.asarray(status_all), np.asarray(survtime_all))

        if verbose > 0:
            print("Epoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

def prediction(model, queryloader, args, testing=False):
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()

    status_all, survtime_all = [], []

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(queryloader):

            bag_tensor, time, status = sampled_batch

            bag_tensor = bag_tensor.to(device)
                
            survtime_all.append(time/30.0)
            status_all.append(status)

        # ===================forward=====================
            y_pred = model(bag_tensor[:,:-2])
            if i_batch == 0:
                y_pred_all = y_pred[:,-1]
            else:
                y_pred_all = torch.cat([y_pred_all, y_pred[:,-1]])

    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)

    loss = _neg_partial_log(y_pred_all, survtime_all, status_all, device)    

    pvalue_pred = cox_log_rank(y_pred_all.data, status_all, survtime_all)
    c_index = calculate_cindex(y_pred_all.data, status_all, survtime_all)

    csv_data = pd.DataFrame({'time':survtime_all,'event':status_all,'risk':np.squeeze(y_pred_all.cpu().numpy())})
    auc = roc(csv_data)

    if not testing:
        print('[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}, auc: {:.3e}'.format(c_index, pvalue_pred, auc))
    else:
        print('[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}, auc: {:.3e}'.format(c_index, pvalue_pred, auc))

        csv_data.to_csv(f'{args.spath_c2 + args.dataset}/risk_{args.fold}.csv') 

    return loss.data.item(), c_index, pvalue_pred, auc

def evaluation(test_df, args):
    test_list = test_df.values.tolist()

    device = torch.device(f'cuda:{args.gpu}')

    model = SurvModel(args.k).to(device)

    dir_pt = f'{args.spath + args.dataset}/inst_fold_{args.fold}/'

    test_data = []
    for data in test_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt')
        except:
            continue
        if fea.shape[0] < args.k: continue
        bag_list = [fea, data[1], data[2]]
        test_data = test_data + [bag_list]

    model.load_state_dict(torch.load(f'{args.spath_c2 + args.dataset}/models/c2_fold_{args.fold}.pth', map_location=device))
    _, c_index, p_value, auc = prediction(model, test_data, args, testing=True) 
    
    with open(f'{args.spath_c2+args.dataset}/result.txt','a')as fr:
        fr.write('fold:'+str(args.fold)+'\n')  
        fr.write('test_ci:'+str(c_index))  
        fr.write('test_p:'+str(p_value)+'\n')  
        fr.write('test_auc:'+str(auc)+'\n')  

    return 