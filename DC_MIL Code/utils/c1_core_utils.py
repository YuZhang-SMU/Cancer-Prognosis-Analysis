import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import pandas as pd
import os
import random
import time
from utils import utils 
from utils.metrics import *
from utils.Early_Stopping import EarlyStopping
from models.model import FirstBranch, SecondBranch, ThirdBranch, get_parameter_number

####################### Curriculum 1 #########################

def train(train_df, val_df, args):
    device = torch.device(f'cuda:{args.gpu}')
    lr=args.lr

    model_save_path = f'{args.spath + args.dataset}/models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_save_path = model_save_path + '/c1_fold_{}.pth'.format(args.fold)

    random.seed(23)
    
    train_list, val_list = train_df.values.tolist(), val_df.values.tolist()
    tile_map = utils.get_tile_map(args.dataset)

    pnum,nnum = 0,0
    train_inst = []
    for data in train_list:
        slideID = data[0]
        class_label = data[-1]
        if class_label == 2: continue

        inst_list = utils.build_inst_multi(tile_map, slideID, class_label, args.sinst, args.sbag, args)
        if class_label == 1:
            pnum += len(inst_list)
        elif class_label == 0:
            nnum += len(inst_list)

        train_inst = train_inst + inst_list
    print(f'training positive dataset = {pnum / (nnum+pnum)}', pnum, nnum)
    weights = torch.FloatTensor([1,pnum / nnum]).to(device)
    
    pnum,nnum = 0,0
    val_inst = []
    for data in val_list:
        slideID = data[0]
        class_label = data[-1]
        if class_label == 2: continue
     
        inst_list = utils.build_inst_multi(tile_map, slideID, class_label, args.sinst, args.sbag, args)       
        if class_label == 1:
            pnum += len(inst_list)
        elif class_label == 0:
            nnum += len(inst_list)
            
        val_inst = val_inst + inst_list
    print(f'test positive dataset = {pnum / (nnum+pnum)}', pnum, nnum)

    model = None
    epoch = 0
    for k, nepoch in enumerate(args.branch_epoch):
        print('loading model')
        args.iter = args.iters[k]          
        n_model = k + 1
        if n_model == 1:
            model = FirstBranch(multi_gpu=args.multi_gpu).to(device)
        elif n_model == 2: 
            model = SecondBranch(model,multi_gpu=args.multi_gpu).to(device)
        elif n_model == 3:
            model = ThirdBranch(model, multi_gpu=args.multi_gpu).to(device)
            
        get_parameter_number(model)    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr[k])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)    
        
        spath = model_save_path.replace('.pth', f'_model_{n_model}.pth')
        # if os.path.exists(spath): model.load_state_dict(torch.load(spath,map_location=device))           
        early_stopping = EarlyStopping(model_path=spath, patience=10, verbose=True)
                     
        for epoch in range(nepoch):
            
            print("\n Branch: {}; Epoch: {}".format(n_model, epoch))

            train_epoch(n_model, model, optimizer, train_inst, args, weights)

            valid_loss, val_acc = prediction(n_model, model, val_inst, args)
            scheduler.step(valid_loss)

            early_stopping(val_acc, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load(spath,map_location=device))

        test_loss, test_acc = prediction(n_model, model, val_inst, args)

    return

def train_epoch(n_model, model, optimizer, train_inst, args, weights=None, measure=1):
    device = torch.device(f'cuda:{args.gpu}')
    model.train()

    y_label = []
    y_patient = {}
    iter = 0
    loss_all = []
    
    begin = time.time()

    random.shuffle(train_inst)
    for i_batch, data in enumerate(tqdm(train_inst)):
        name = data[-1]
        if n_model == 1:
            x, label = utils.data_load_single(data, device)
        elif n_model == 2:
            x, label = utils.data_load_double(data, device)
        else:
            x, label = utils.data_load_multi(data, device)

        y_label.append(label.numpy())
        # ===================forward=====================
        y_prob, y_hat = model(x)
        y_hat = y_hat.detach().cpu().numpy()
        
        if name not in y_patient.keys():
            y_patient[name] = []
        y_patient[name].append(y_hat)

        if i_batch == 0:
            y_hat_all = y_hat
        else:
            y_hat_all = np.concatenate([y_hat_all, y_hat])

        if iter == 0:
            y_each = label
            y_prob_each = y_prob
        else:
            y_each = torch.cat([y_each, label])
            y_prob_each = torch.cat([y_prob_each, y_prob])
        iter += 1

        if iter % args.iter == 0 or i_batch == len(train_inst)-1:
            optimizer.zero_grad()

            loss_surv = loss_fn(y_prob_each, y_each.to(device), weights)

            l_reg = torch.tensor(0.).to(device)
            for W in model.parameters():
                l_reg += torch.norm(W)

            l_ht = 0
            if n_model > 1:
                for (p1,p2) in zip(model.FT_param_ori, model.FT_param):
                    l_ht += torch.norm(p1-p2)

            loss = loss_surv + args.l2w * l_reg + args.beta_o * l_ht
        # ===================backward====================
            loss.backward()
            optimizer.step()

            y_prob_each = None
            y_each = []
            loss_all.append(loss.data.item())
            iter = 0
            
            # torch.cuda.empty_cache()
            # gc.collect()

    end = time.time()
    print('time: {}s'.format(end-begin))

    if measure:
        num = len(y_label)
        y_label = np.asarray(y_label)
        train_acc = eval_ans(y_hat_all, y_label.reshape(-1)) / num 
        loss_all = np.mean(loss_all,0)
        total_train_acc = get_total_acc(train_inst, y_patient)
        print(['training_loss: {:.4f}'.format(loss_all)])
        print(['acc: {:.4f}'.format(train_acc), 'total_acc: {:.4f}'.format(total_train_acc)])
        
    return

def prediction(n_model, model, queryloader, args, verbose=1):
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()
    y_patient = {}
    tbar = tqdm(queryloader, desc='\r')
    with torch.no_grad():
        for i_batch, data in enumerate(tbar):
            name = data[-1]

            if n_model == 1:
                x, label = utils.data_load_single(data, device)
            elif n_model == 2:
                x, label = utils.data_load_double(data, device)
            elif n_model == 3:
                x, label = utils.data_load_multi(data, device)

            # ===================forward=====================
            y_prob, y_hat = model(x)

            y_hat = y_hat.detach().cpu().numpy()
            
            if name not in y_patient.keys():
                y_patient[name] = []
            y_patient[name].append(y_hat)

            if i_batch == 0:
                y_label = label
                y_hat_all = y_hat
                y_prob_all = y_prob
            else:
                y_label = torch.cat([y_label, label])
                y_hat_all = np.concatenate([y_hat_all, y_hat])
                y_prob_all = torch.cat([y_prob_all, y_prob])
        
    num = i_batch+1

    loss_surv = loss_fn(y_prob_all, y_label.to(device))

    l_reg = torch.tensor(0.).to(device)
    for W in model.parameters():
        l_reg += torch.norm(W)

    l_ht = 0
    if n_model > 1:
        for (p1,p2) in zip(model.FT_param_ori, model.FT_param):
            l_ht += torch.norm(p1-p2)

    loss = loss_surv + args.l2w * l_reg + args.beta_o * l_ht
                                    
    y_label = np.asarray(y_label)
    
    acc = eval_ans(y_hat_all, y_label.reshape(-1)) / num 
    total_acc = get_total_acc(queryloader, y_patient)
    
    if verbose:
        print(['loss: {:.4f}'.format(loss)])
        print(['acc: {:.4f}'.format(acc), 'total_acc: {:.4f}'.format(total_acc)])
    
    return loss.data.item(), total_acc

def inst_encoding(train_df, test_df, args):
    random.seed(23)
    device = torch.device(f'cuda:{args.gpu}')
    model_save_path = args.spath+'{}/models/c1_fold_{}.pth'.format(args.dataset,args.fold)
    tile_map = utils.get_tile_map(args.dataset)

    print('loading model')
    model = FirstBranch(multi_gpu=0).to(device)
    model.load_state_dict(torch.load(model_save_path.replace('.pth', '_model_1.pth'),map_location=device))
    model = SecondBranch(model,multi_gpu=0).to(device)
    model.load_state_dict(torch.load(model_save_path.replace('.pth', '_model_2.pth'),map_location=device))
    model = ThirdBranch(model,multi_gpu=0).to(device)
    model.load_state_dict(torch.load(model_save_path.replace('.pth', '_model_3.pth'),map_location=device))

    train_list, test_list = train_df.values.tolist(), test_df.values.tolist()
    train_res = []
    for data in train_list:
        slideID = data[0]
        class_label = data[-1]
        inst_list = utils.build_inst_multi(tile_map, slideID, class_label, args.sinst, 10, args)
        if not len(inst_list):
            print(slideID)
            continue
        train_res.append(data+encoder_pt(model, inst_list, args))
    
    test_res = []
    for data in test_list:
        slideID = data[0]
        class_label = data[-1]
        inst_list = utils.build_inst_multi(tile_map, slideID, class_label, args.sinst, 10, args)
        if not len(inst_list):
            print(slideID)
            continue
        test_res.append(data+encoder_pt(model, inst_list, args))

    train_res = pd.DataFrame(data=train_res)
    test_res = pd.DataFrame(data=test_res)

    res_path = args.spath+'{}/'.format(args.dataset)
    writer = pd.ExcelWriter(res_path+'fold_{}.xlsx'.format(args.fold))
    train_res.to_excel(writer, sheet_name='train', index=False)
    test_res.to_excel(writer, sheet_name='test', index=False)
    writer.save()
    writer.close()

    return

def encoder_pt(model, queryloader, args):
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()
    save_path =  args.spath+'{}/inst_fold_{}/'.format(args.dataset,args.fold)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    name = queryloader[0][-1]
    class_label = queryloader[0][-2]
    save_path += f'{name}.pt'

    features, pres, outs = [], [], 0
    for data in queryloader:
        x, _ = utils.data_load_multi(data, device)
        f, pre, out = model._fea_map(x) 
        
        features.append(torch.cat((f,out.view(1,-1)),1))
        pres.append(pre)
        outs += out.numpy()
    res = 1 if (outs / len(queryloader)) >= 0.5 else 0

    if class_label == 2:
        class_label = res

    loss = [F.cross_entropy(pre, torch.tensor([class_label])) for pre in pres]
    loss = torch.FloatTensor(loss).unsqueeze(1)

    features = torch.cat(features,0)
    fea_and_loss = torch.cat((features,loss),1)
    torch.save(fea_and_loss,save_path)
    return [res]
