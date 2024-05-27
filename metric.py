import torch
import numpy as np
import math
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

# Compute loss
def compute_EXP_loss(pred, label, weights):
    cri_exp = nn.CrossEntropyLoss(weights)
    cls_loss = cri_exp(pred, label)
    return cls_loss

def compute_AU_loss(pred, label, weights):
    cri_AU = nn.BCEWithLogitsLoss(weights)
    cls_loss = cri_AU(pred, label.float())
    return cls_loss

def CCC_loss(x, y):
    x, y = x.view(-1), y.view(-1)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m, y_m = torch.mean(x), torch.mean(y)
    x_s, y_s = torch.std(x), torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1-ccc

def compute_VA_loss(Vout,Aout,label):
    ccc_loss = CCC_loss(Vout[:,0],label[:,0]) + CCC_loss(Aout[:,0],label[:,1])
    mse_loss = nn.MSELoss()(Vout,label[:,0]) + nn.MSELoss()(Aout,label[:,1])
    return mse_loss,ccc_loss

# Compute F1 score
def compute_EXP_F1(pred, target):
    pred_labels = np.argmax(pred, axis=1)
    target_labels = np.argmax(target, axis=1)
    macro_f1 = f1_score(target_labels,pred_labels,average='macro')
    acc = accuracy_score(target_labels, pred_labels)
    return macro_f1, acc

def f1s_max_AU(label, pred, thresh, i=0):
    pred = np.array(pred)
    label = np.array(label)
    label = label[:,i]
    pred = pred[:,i]
    acc = []
    F1 = []
    for i in thresh:
        new_pred = ((pred >= i) * 1).flatten()
        acc.append(accuracy_score(label.flatten(), new_pred))
        F1.append(f1_score(label.flatten(), new_pred))

    F1_MAX = max(F1)
    if F1_MAX < 0 or math.isnan(F1_MAX):
        F1_MAX = 0
        F1_THRESH = 0
        accuracy = 0
    else:
        idx_thresh = np.argmax(F1)
        F1_THRESH = thresh[idx_thresh]
        accuracy = acc[idx_thresh]
    return F1, F1_MAX, F1_THRESH, accuracy

def compute_AU_F1(pred,label,thresh=np.arange(0.1,1,0.1)):
    F1s = []
    F1t = []
    acc = []
    for i in range(12):
        F1, F1_MAX, F1_THRESH, accuracy = f1s_max_AU(label,pred,thresh,i)
        F1s.append(F1_MAX)
        F1t.append(F1_THRESH)
        acc.append(accuracy)
    acc = [round(a,3) for a in acc]
    return np.mean(F1s),np.mean(F1t),acc, F1t

# Concordance Correlation Coefficient
def CCC_score(x, y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def compute_VA_CCC(x,y):
    x = np.array(x)
    y = np.array(y)
    x[x>1] = 1
    x[x<-1] = -1
    ccc1 = CCC_score(x[:,0],y[:,0])
    ccc2 = CCC_score(x[:,1],y[:,1])

    return ccc1,ccc2
