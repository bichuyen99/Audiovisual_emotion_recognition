import os
import torch
from tqdm import tqdm
import time
import numpy as np
import pickle
import torch.nn as nn
from sklearn.metrics import f1_score
import random
import warnings
from metric import compute_VA_CCC
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = '/MSc/Thesis'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

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
        return np.array([p.cpu().detach().numpy() for p in preds_proba])

def slide_window(preds_proba, i, delta, typ):
    i1 = max(i - delta, 0)
    if typ == 'mean':
        proba = np.mean(preds_proba[i1:i+delta+1], axis=0)
    elif typ == 'median':
        proba = np.median(preds_proba[i1:i+delta+1], axis=0)
    else:
        proba = np.mean(preds_proba[i1:i+delta+1:int(typ)], axis=0)
    return np.argmax(proba), proba

# Challenges
task = ['EXPR_Recognition_Challenge','AU_Detection_Challenge','VA_Estimation_Challenge']
split = ['Train_Set', 'Validation_Set']
typ = ['Train','Val','Test']
vis_typ = ['cropped_aligned', 'cropped']
visual_feat = 'visualfeat_enet_b2_8_best'
audio_feat = ['audiofeat_wav2vec2','audiofeat_vggish','nope']
vis_aud = ['visual_wav2vec2','visual_vggish','visual']
batch_size = 32
model_type = ['fusion', 'mlp']
vis = vis_typ[0]
viau = vis_aud[0]
auft = audio_feat[0]

# EXPR
tsk = task[0]
class MLPModel(nn.Module):
    def __init__(self, audio_ft = auft, num_classes=8):
        super(MLPModel, self).__init__()
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.activ = nn.ReLU()
        self.fc1 = nn.Linear(self.concat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, vis_feat, aud_feat):
        inputs = [vis_feat]
        inputs.append(aud_feat)
        feat = torch.cat(inputs, dim=0)
        feat = self.fc1(feat)
        feat = self.activ(feat)
        out = self.fc2(feat)
        return out, torch.softmax(out, dim=0)

mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_EXPR_mlp_{viau}.pth'))
mlp_model = MLPModel().to(device)
mlp_model.load_state_dict(mlp_best_model)

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

hyperparams=[(isMean,delta) for delta in [0, 15, 30, 60, 100, 200] for isMean in [1,0] if not (isMean==0 and delta==0)]
total_true=[]
total_preds=[[] for _ in range(len(hyperparams))]
timing_results = {(isMean, delta): 0 for isMean, delta in hyperparams}
for videoname,(img, predict, label) in test_vid.items():
    for i,ind in enumerate(img):
        total_true.append(label[i].cpu().numpy())
    preds_proba = smooth_prediction(img, predict)
    for hInd,(isMean,delta) in enumerate(hyperparams):
        preds=[]
        start = time.time()
        for i in range(len(preds_proba)):
            i1=max(i-delta,0)
            if isMean:
                best_ind, proba = slide_window(preds_proba, i, delta, 'mean')
            else:
                best_ind, proba = slide_window(preds_proba, i, delta, 'median')
            preds.append(best_ind)
        for i,ind in enumerate(img):
            if label[i]>=0:
                total_preds[hInd].append(preds[ind-1])
        end = time.time()
        timing_results[(isMean, delta)] += end - start
total_true=np.array(total_true)

for hInd, (isMean, delta) in enumerate(hyperparams):
    preds = np.array(total_preds[hInd])
    accuracy = round((preds == total_true).mean(), 3)
    f1 = round(f1_score(y_true=total_true, y_pred=preds, average='macro'), 3)
    mean_or_median = 'mean' if isMean else 'median'
    time_taken = round(timing_results[(isMean, delta)],3)
    print(f'{mean_or_median}; delta: {delta}; Acc: {accuracy}; F1: {f1}; Time: {time_taken}s')

# AU
tsk = task[1]

class MLPModel(nn.Module):
    def __init__(self, audio_ft = auft, num_classes=12):
        super(MLPModel, self).__init__()
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.activ = nn.ReLU()
        self.fc1 = nn.Linear(self.concat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs, dim=0)
        feat = self.fc1(feat)
        feat = self.activ(feat)
        out = self.fc2(feat)
        return out, torch.sigmoid(out)

mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_AU_mlp_{viau}.pth'))
mlp_model = MLPModel().to(device)
mlp_model.load_state_dict(mlp_best_model)

anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[1]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[2]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/AU/{tsk}_{typ[2]}_{viau}.pkl'), 'rb') as f:
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

thresholds = np.array([0.30000000000000004, 0.2, 0.2, 0.30000000000000004, 0.5, 0.30000000000000004, 0.6, 0.1, 0.1, 0.1, 0.6, 0.30000000000000004])

hyperparams=[(isMean,delta) for delta in [0, 15, 30, 60, 100, 200] for isMean in [1,0] if not (isMean==0 and delta==0)]
total_true=[]
total_preds=[[] for _ in range(len(hyperparams))]
timing_results = {(isMean, delta): 0 for isMean, delta in hyperparams}
for videoname,(img, predict, label) in test_vid.items():
    for i,ind in enumerate(img):
        total_true.append(label[i].cpu().numpy())
    preds_proba = smooth_prediction(img, predict)
    for hInd,(isMean,delta) in enumerate(hyperparams):
        preds=[]
        start = time.time()
        for i in range(len(preds_proba)):
            i1=max(i-delta,0)
            if isMean:
                _, proba = slide_window(preds_proba, i, delta, 'mean')
            else:
                _, proba = slide_window(preds_proba, i, delta, 'median')
            aus = (proba>=thresholds)*1
            preds.append(aus)
        for i,ind in enumerate(img):
            if label[i][0]>=-1 and label[i][1]>=-1:
                total_preds[hInd].append(preds[ind-1])
        end = time.time()
        timing_results[(isMean, delta)] += end - start
total_true=np.array(total_true)

for hInd, (isMean, delta) in enumerate(hyperparams):
    preds = np.array(total_preds[hInd])
    accuracy = round((preds == total_true).mean(), 3)
    f1 = round(np.mean([f1_score(y_true=total_true[:,i],y_pred=preds[:,i]) for i in range(preds.shape[1])]), 3)
    mean_or_median = 'mean' if isMean else 'median'
    time_taken = round(timing_results[(isMean, delta)],3)
    print(f'{mean_or_median}; delta: {delta}; Acc: {accuracy}; F1: {f1}; Time: {time_taken}s')

# VA
tsk = task[2]

class MLPModel(nn.Module):
    def __init__(self, audio_ft = auft, num_classes=1):
        super(MLPModel, self).__init__()
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.activ = nn.ReLU()
        self.fc1 = nn.Linear(self.concat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs, dim=0)
        feat = self.fc1(feat)
        feat = self.activ(feat)
        vout = self.fc2(feat)
        aout = self.fc2(feat)

        return vout, aout, torch.tanh(vout), torch.tanh(aout)

mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_VA_model/best_VA_mlp_{viau}.pth'))
mlp_model = MLPModel().to(device)
mlp_model.load_state_dict(mlp_best_model)

anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[1]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[2]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/VA/{tsk}_{typ[2]}_{viau}.pkl'), 'rb') as f:
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

hyperparams=[(isMean,delta) for delta in [0, 15, 30, 60, 100, 200] for isMean in [1,0] if not (isMean==0 and delta==0)]
total_true=[]
total_preds=[[] for _ in range(len(hyperparams))]
timing_results = {(isMean, delta): 0 for isMean, delta in hyperparams}
for videoname,(img, predict, label) in test_vid.items():
    for i,ind in enumerate(img):
        total_true.append(label[i].cpu().numpy())
    preds_proba = smooth_prediction(img, predict)
    for hInd,(isMean,delta) in enumerate(hyperparams):
        preds=[]
        start = time.time()
        for i in range(len(preds_proba)):
            i1=max(i-delta,0)
            if isMean:
                best_ind, proba = slide_window(preds_proba, i, delta, 'mean')
            else:
                best_ind, proba = slide_window(preds_proba, i, delta, 'median')
            preds.append(proba)
        for i, ind in enumerate(img):
            if label[i][0]>=-1 and label[i][1]>=-1:
                total_preds[hInd].append(preds[ind-1])
        end = time.time()
        timing_results[(isMean, delta)] += end - start
total_true=np.array(total_true)

for hInd, (isMean, delta) in enumerate(hyperparams):
    preds = np.array(total_preds[hInd])
    ccc1, ccc2 = compute_VA_CCC(preds, total_true)
    mean_or_median = 'mean' if isMean else 'median'
    time_taken = round(timing_results[(isMean, delta)],3)
    print(f'{mean_or_median}; delta: {delta}; CCCV: {ccc1:.3f}; CCCA: {ccc2:.3f}; Time: {time_taken}s')