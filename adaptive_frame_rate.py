import os
import torch
from tqdm import tqdm
import time
import numpy as np
import pickle
import random
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/MSc/Thesis'

def smooth_prediction(img, predict):
    cur_ind = 0
    preds_proba = []
    if img:
        for i in range(img[-1]):
            if img[cur_ind] - 1 == i:
                preds_proba.append(predict[cur_ind])
                cur_ind += 1
            else:
                if cur_ind == 0:
                    preds_proba.append(predict[cur_ind])
                else:
                    w = (i - img[cur_ind - 1] + 1) / (img[cur_ind] - img[cur_ind - 1])
                    pred = w * predict[cur_ind - 1] + (1 - w) * predict[cur_ind]
                    preds_proba.append(pred)
        try:
            preds_proba = np.array([p.cpu().detach().numpy() for p in preds_proba])
        except:
            preds_proba = np.array(preds_proba)
        return preds_proba

def slide_window(preds_proba, i, delta, typ):
    i1 = max(i - delta, 0)
    if typ == 'mean':
        proba = np.mean(preds_proba[i1:i+delta+1], axis=0)
    elif typ == 'median':
        proba = np.median(preds_proba[i1:i+delta+1], axis=0)
    else:
        proba = np.median(preds_proba[i1:i+delta+1:int(typ)], axis=0)
    return np.argmax(proba), proba

# EXPR

delta = 200
tsk = task[0]
vis = vis_typ[0]
viau = vis_aud[0]
auft = audio_feat[0]

anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[0]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[0]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/EXPR/{tsk}_{typ[0]}_{viau}.pkl'), 'rb') as f:
    data = pickle.load(f)
train_vid = {}
for vname in tqdm(vidnames):
        img, predict, label = [], [], []
        for imgname, val in sorted(data[tsk][vname].items()):
            vis_feat = torch.tensor(val[visual_feat]).to(device)
            if auft == 'nope':
                aud_feat = None
            else:
                aud_feat = torch.tensor(val[auft]).to(device)
            if tsk == task[2]:
                _, _, vpred, apred = mlp_model(vis_feat, aud_feat)
                preds = torch.tensor([vpred, apred])
            else:
                _, pred = mlp_model(vis_feat, aud_feat)
                preds = torch.tensor(pred)
            ind = int(imgname.split('/')[1][:-4])
            img.append(ind)
            predict.append(preds)
            label.append(data[tsk][vname][imgname]['label'])
        train_vid[vname] = (img, predict, label)

anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[1]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[2]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/EXPR/{tsk}_{typ[2]}_{viau}.pkl'), 'rb') as f:
    data = pickle.load(f)

test_vid = {}
for vname in tqdm(vidnames):
        img, predict, label = [], [], []
        for imgname, val in sorted(data[tsk][vname].items()):
            vis_feat = torch.tensor(val[visual_feat]).to(device)
            if auft == 'nope':
                aud_feat = None
            else:
                aud_feat = torch.tensor(val[auft]).to(device)
            if tsk == task[2]:
                _, _, vpred, apred = mlp_model(vis_feat, aud_feat)
                preds = torch.tensor([vpred, apred])
            else:
                _, pred = mlp_model(vis_feat, aud_feat)
                preds = torch.tensor(pred)
            ind = int(imgname.split('/')[1][:-4])
            img.append(ind)
            predict.append(preds)
            label.append(data[tsk][vname][imgname]['label'])
        test_vid[vname] = (img, predict, label)

stride2scores={}
for stride in [200, 100, 50, 25, 10]:
    total_true, predictions, max_decision_values = [],[],[]
    for vidname, (img, predict, label) in train_vid.items():
        index = []
        for i,ind in enumerate(img):
            total_true.append(label[i].cpu().numpy())
            index.append(ind-1)
        preds_proba = smooth_prediction(img, predict)
        for i in range(len(index)):
            best_ind, proba = slide_window(preds_proba, index[i], delta, stride)
            predictions.append(best_ind)
            max_decision_values.append(proba[best_ind])
    stride2scores[stride] = (np.array(total_true),np.array(predictions),np.array(max_decision_values))

def get_threshold(stride,fpr_corrected):
    (total_true,predictions,max_decision_values) = stride2scores[stride]
    mistakes = max_decision_values[predictions != total_true]
    best_threshold = -1
    for i, threshold in enumerate(sorted(max_decision_values[predictions == total_true])[::-1]):
        tpr = i/len(predictions)
        fpr = (mistakes > threshold).sum()/len(predictions)

        if fpr > fpr_corrected:
            if best_threshold == -1:
                best_threshold = threshold
            print(stride, 'best_threshold', best_threshold, i)
            break
        best_threshold = threshold
    return best_threshold

stride2threshold = {}
for stride in stride2scores:
    fpr_corrected=0.05
    stride2threshold[stride] = get_threshold(stride,fpr_corrected)
stride2threshold[1] = 0
print(stride2threshold)

all_strides=[
    [200, 100, 50, 10, 1],
    [50, 25, 1],
    [50, 10, 1],
    [200,50,1],
    [100,50,1],
    [200,1],
    [100,1],
    [50,1]
]
for s in stride2threshold.keys():
    all_strides.append([s])

for strides in all_strides:
    print(strides)
    last_stride=strides[-1]

    total_true=[]
    total_preds=[]
    total_frames_processed,total_frames=0,0
    time_each = []
    start = time.time()
    for videoname, (img, predict, label) in test_vid.items():
        emotional_img=[]
        start1 = time.time()
        for i,ind in enumerate(img):
            total_true.append(label[i].cpu().numpy())
            emotional_img.append(ind-1)
        cur_ind=0
        preds_proba=[]
        for i in range(img[-1]):
            if img[cur_ind]-1==i:
                preds_proba.append(predict[cur_ind])
                cur_ind+=1
            else:
                if cur_ind==0:
                    preds_proba.append(predict[cur_ind])
                else:
                    w=(i-img[cur_ind-1]+1)/(img[cur_ind]-img[cur_ind-1])
                    pred=w*predict[cur_ind-1]+(1-w)*predict[cur_ind]
                    preds_proba.append(pred)

        preds_proba=np.array([p.cpu().numpy() for p in preds_proba])

        preds=-np.ones(len(emotional_img))
        end1 = time.time()
        time_each.append(end1 - start1)
        for stride in strides:
            threshold=stride2threshold[stride]
            for i in range(len(emotional_img)):
                if preds[i]<0:
                    i1=max(emotional_img[i]-delta,0)
                    cur_preds=preds_proba[i1:emotional_img[i]+delta+1:stride]
                    proba=np.median(cur_preds,axis=0)
                    best_ind=np.argmax(proba)
                    if proba[best_ind]>=threshold or stride==last_stride:
                        total_frames_processed+=len(cur_preds)
                        total_frames+=len(preds_proba[i1:emotional_img[i]+delta+1])
                        preds[i]=best_ind
        for p in preds:
            total_preds.append(p)
    end = time.time()
    elapsed_time = end - start - sum(time_each)
    total_true=np.array(total_true)
    preds=np.array(total_preds)
    print('Acc:',round((preds==total_true).mean(),3), 'F1:',round(f1_score(y_true=total_true,y_pred=preds, average="macro"),3))
    print(total_frames_processed,total_frames,round(total_frames_processed/total_frames,3))
    print(f"Time: {elapsed_time:.2f} seconds")
