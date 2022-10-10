import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import torch
from sklearn.metrics import roc_curve, auc


def _neg_partial_log(prediction, T, E, device):
    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.to(device)
    train_ystatus = torch.FloatTensor(E).to(device)

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn

def roc(data):

    _list = data.values.tolist()
    T = 3 * 12
    k = 0
    _copy = _list[:]
    for i in _copy:
        if i[0] <= T and i[1]==0:
            _list.pop(k) 
        else:
            k = k + 1
    _list = np.array(_list)
    y = (_list[:,0] < T).astype('int')
    status = _list[:,1]
    x = _list[:,2]

    fpr, tpr, thresholds = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def loss_fn(class_prob, class_label, weights=None):
    loss = torch.nn.CrossEntropyLoss(weight=weights)
    return loss(class_prob, class_label)

def eval_ans(y_hat, true_label):
    return sum(y_hat == true_label)

def get_total_acc(train_data, y_patient):
    n = 0
    n_total = 0
    for data in train_data:
        name = data[-1]
        label = data[-2]
        if name in y_patient:
            k_true, k_total = 0, 0
            for ys in y_patient[name]:
                if ys == label:
                    k_true += 1
                k_total += 1
            if k_true / k_total >= 0.5:
                n += 1
            n_total += 1
    acc = n/n_total
    return acc

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all
    idx = hazards_dichotomize == 0
    labels = labels
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def calculate_cindex(hazards, labels, survtime_all):
    labels = labels
    hazards = hazards.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))
