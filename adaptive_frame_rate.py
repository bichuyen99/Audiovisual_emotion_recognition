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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

task = ['EXPR_Recognition_Challenge','AU_Detection_Challenge','VA_Estimation_Challenge']
split = ['Train_Set', 'Validation_Set']
typ = ['Train','Val','Test']
vis_typ = ['cropped_aligned', 'cropped']
visual_feat = 'visualfeat_enet_b2_8_best'
audio_feat = ['audiofeat_wav2vec2','audiofeat_vggish','nope']
vis_aud = ['visual_wav2vec2','visual_vggish','visual']
batch_size = 32
model_type = ['fusion', 'mlp']

# EXPR

delta = 100

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
                    proba=np.mean(cur_preds,axis=0)
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

# AU

delta = 15

anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[0]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[0]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/AU/{tsk}_{typ[0]}_visual.pkl'), 'rb') as f:
        data = pickle.load(f)

task1 = task[1]
feature_a = 'audiofeat_wav2vec2'
feat_root = os.path.join(root + '/models/ABAW6', feature_a)
filenames = os.listdir(feat_root)[:]
for vname in tqdm(vidnames):
    feature = np.load(os.path.join(feat_root, f'{vname}.npy'), allow_pickle=True).tolist()
    for imgname, val in feature.items():
        if imgname in data[task1][vname]:
            data[task1][vname][imgname].update({f'{feature_a}': val})
        for img, value in list(data[task1][vname].items()):
            if len(value) < 3:
                data[task1][vname].pop(img)

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

thresholds = np.array([0.30000000000000004, 0.2, 0.2, 0.30000000000000004, 0.5, 0.30000000000000004, 0.6, 0.1, 0.1, 0.1, 0.6, 0.30000000000000004])

stride2scores={}
for stride in [200, 100, 50, 25, 10]:
    total_true, predictions, max_decision_values = [],[],[]
    decision_values = [[] for _ in range(8)]
    for vidname, (img, predict, label) in train_vid.items():
        index = []
        for i,ind in enumerate(img):
            total_true.append(label[i].cpu().numpy())
            index.append(ind-1)
        preds_proba = smooth_prediction(img, predict)
        for i in range(len(index)):
            best_ind, proba = slide_window(preds_proba, index[i], delta, stride)
            aus = (proba>=thresholds)*1
            predictions.append(aus)
            for j in range(8):
                if proba[j] > thresholds[j]:
                    decision_values[j].append(proba[j])
                else:
                    decision_values[j].append(thresholds[j])
            max_decision_values.append([max(values) for values in decision_values])
    stride2scores[stride] = (np.array(total_true),np.array(predictions),np.array(max_decision_values))

def get_threshold(stride,fpr_corrected):
    (total_true,predictions,max_decision_values) = stride2scores[stride]
    mistakes = max_decision_values[predictions.any() != total_true.any()]
    best_threshold = -1
    for i, threshold in enumerate(sorted(max_decision_values[predictions.all() == total_true.all()])[::-1]):
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
    start = time.time()
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
        preds=[[-1]*12 for _ in range(len(emotional_img))]
        end1 = time.time()
        time_each.append(end1-start1)
        for stride in strides:
            threshold=stride2threshold[stride]
            for i in range(len(emotional_img)):
                if max(preds[i])<0:
                    i1=max(emotional_img[i]-delta,0)
                    cur_preds=preds_proba[i1:emotional_img[i]+delta+1:stride]
                    proba=np.mean(cur_preds,axis=0)
                    aus=(proba>=thresholds)*1
                    if stride != 1:
                        if proba.all()>=threshold.all() or stride==last_stride:
                            total_frames_processed+=len(cur_preds)
                            total_frames+=len(preds_proba[i1:emotional_img[i]+delta+1])
                            preds[i] = aus
                    else:
                        if proba.all()>=threshold or stride==last_stride:
                            total_frames_processed+=len(cur_preds)
                            total_frames+=len(preds_proba[i1:emotional_img[i]+delta+1])
                            preds[i] = aus
        for p in preds:
            total_preds.append(p)
    end = time.time()
    elapsed_time = end - start - sum(time_each)
    total_true=np.array(total_true)
    pred=np.array(total_preds)
    print('Acc:',round((pred==total_true).mean(),3), 'F1:',round(np.mean([f1_score(y_true=total_true[:,i],y_pred=pred[:,i]) for i in range(pred.shape[1])]),3))
    print(total_frames_processed,total_frames,round(total_frames_processed/total_frames,3))
    print(f"Time: {elapsed_time:.2f} seconds")

# VA
anno_path = os.path.join(root,f'data/Annotations/{tsk}/{split[0]}')
with open(os.path.join(root, f'data/Annotations/{tsk}/{typ[0]}.txt'), 'r') as f:
    vidnames = f.read().splitlines()
with open(os.path.join(root,f'models/ABAW6/{vis}/VA/{tsk}_{typ[0]}_visual.pkl'), 'rb') as f:
        data = pickle.load(f)

task1 = task[2]
feature_a = 'audiofeat_wav2vec2'
feat_root = os.path.join(root + '/models/ABAW6', feature_a)
filenames = os.listdir(feat_root)[:]
for vname in tqdm(vidnames):
    feature = np.load(os.path.join(feat_root, f'{vname}.npy'), allow_pickle=True).tolist()
    for imgname, val in feature.items():
        if imgname in data[task1][vname]:
            data[task1][vname][imgname].update({f'{feature_a}': val})
        for img, value in list(data[task1][vname].items()):
            if len(value) < 3:
                data[task1][vname].pop(img)

img, predict, label = [], [], []
vid = {}
iterator = iter(train_loader)
for i in range(len(train_loader)//32):
    VA = next(iterator)
    images = VA['frame']
    vis_feat, aud_feat, y = VA[visual_feat], VA[auft], VA['label']
    vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
    _, _, vpred, apred = mlp_model(vis_feat, aud_feat)
    preds = torch.cat((vpred, apred), dim=1)
    img.extend(images)
    predict.extend(preds)
    label.extend(y)

index, pred, lab = [], [], []
train_vid = {}
for i, val in enumerate(img):
    ind = int(val.split('/')[1][:-4])
    vname = val.split('/')[0]
    if i == 0:
        prename = vname
        index.append(ind)
        pred.append(predict[i])
        lab.append(label[i])
    else:
        if vname == prename:
            index.append(ind)
            pred.append(predict[i])
            lab.append(label[i])
        else:
            combined = list(zip(index, pred, lab))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            index_list_sorted, pred_list_sorted, lab_list_sorted = zip(*combined_sorted)
            train_vid[prename] = (list(index_list_sorted), list(pred_list_sorted), list(lab_list_sorted))
            prename = vname
            index, pred, lab = [], [], []
            index.append(ind)
            pred.append(predict[i])
            lab.append(label[i])

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

delta = 30

stride2scores={}
for stride in [200, 100, 50, 25, 10]:
    total_true, predictions, max_decision_values, decision_values1, decision_values2 = [],[],[], [], []
    for vidname, (img, predict, label) in train_vid.items():
        index = []
        for i,ind in enumerate(img):
            total_true.append(label[i].cpu().numpy())
            index.append(ind-1)
        preds_proba = smooth_prediction(img, predict)
        for i in range(len(index)):
            best_ind, proba = slide_window(preds_proba, index[i], delta, stride)
            predictions.append(proba)
            decision_values1.append(proba[0])
            decision_values2.append(proba[1])
            max_decision_values.append([max(decision_values1),max(decision_values2)])
    stride2scores[stride] = (np.array(total_true),np.array(predictions),np.array(max_decision_values))

def get_threshold(stride,fpr_corrected):
    (total_true,predictions,max_decision_values) = stride2scores[stride]
    mistakes = max_decision_values[predictions.any() != total_true.any()]
    best_threshold = -1
    for i, threshold in enumerate(sorted(max_decision_values[predictions.any() == total_true.any()])[::-1]):
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

        preds_proba=np.array([p.cpu().detach().numpy() for p in preds_proba])

        preds=[[1]*2 for _ in range(len(emotional_img))]
        end1 = time.time()
        time_each.append(end1 - start1)
        for stride in strides:
            threshold=stride2threshold[stride]
            for i in range(len(emotional_img)):
                if preds[i][0]>=-1 and preds[i][1]>=-1:
                    i1=max(emotional_img[i]-delta,0)
                    cur_preds=preds_proba[i1:emotional_img[i]+delta+1:stride]
                    proba=np.mean(cur_preds,axis=0)
                    best_ind=np.argmax(proba)
                    if stride == 1:
                        if proba[best_ind]>=threshold or stride==last_stride:
                                total_frames_processed+=len(cur_preds)
                                total_frames+=len(preds_proba[i1:emotional_img[i]+delta+1])
                                preds[i]=proba
                    else:
                        if proba[0].all()>=threshold[0].all() or proba[1].all()>=threshold[1].all() or stride==last_stride:
                                total_frames_processed+=len(cur_preds)
                                total_frames+=len(preds_proba[i1:emotional_img[i]+delta+1])
                                preds[i]=proba

        for p in preds:
            total_preds.append(p)
    end = time.time()
    elapsed_time = end - start - sum(time_each)
    total_true=np.array(total_true)
    preds=np.array(total_preds)
    ccc1, ccc2 = compute_VA_CCC(preds, total_true)
    print(f'CCCV: {ccc1:.3f}; CCCA: {ccc2:.3f}')
    print(total_frames_processed,total_frames,round(total_frames_processed/total_frames,3))
    print(f"Time: {elapsed_time:.2f} seconds")