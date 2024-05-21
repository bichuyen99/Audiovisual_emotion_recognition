import os
import torch
import numpy as np
import random
from torch import optim
from sklearn.utils.class_weight import compute_class_weight
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

# EXPR Recognition Challenge
EXP_model = EXP_fusion().to(device)
mlp_model = MLPModel().to(device)

def one_hot_transfer(label, class_num):
    one_hot = torch.eye(class_num)
    one_hot = one_hot.to(device)
    return one_hot[label]

y_train = []
iterator = iter(train_loader)
i = 0
while True:
    try:
        EXPR = next(iterator)
        y_train.extend(EXPR['label'].numpy())
    except:
        break

class_weights=compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
weights=torch.tensor(class_weights,dtype=torch.float).to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, EXP_model.parameters()), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, mlp_model.parameters()), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

# Training
def train(model, mod_type, train_loader, val_loader, epoch, batch_size, optim, au_feat, weight, vi_au):

    model.train(True)
    model.eval()
    best_loss = float('inf')
    f1best, accbest = 0, 0
    loss_value = []
    loss_train = []
    loss_val = []
    all_preds = []
    all_targets = []

    for e in range(epoch):
        print(f'Epoch: {e+1}')
        iterator = iter(train_loader)
        while True:
            try:
                EXPR = next(iterator)
                if au_feat == 'nope':
                    vis_feat, y = EXPR[visual_feat], EXPR['label']
                    vis_feat, y = vis_feat.to(device), y.to(device)
                    aud_feat = None
                else:
                    vis_feat, aud_feat, y = EXPR[visual_feat], EXPR[au_feat], EXPR['label']
                    vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
                y_onehot = one_hot_transfer(y, 8).to(device)
                model.zero_grad()
                pred, exp_pred = model(vis_feat, aud_feat)
                loss = compute_EXP_loss(pred, y_onehot, weight)
                loss.backward()
                optim.step()
                loss_value.append(loss.item())
                all_preds.extend(exp_pred.cpu().tolist())
                all_targets.extend(y_onehot.cpu().tolist())
            except:
                break
        avg_loss = round(np.mean(loss_value),3)
        loss_train.append(avg_loss)
        f1_scores, accuracy = compute_EXP_F1(all_preds, all_targets)
        print(f'Train Loss: {avg_loss}, Accuracy: {round(accuracy,3)}')

        val_loss, f1s, acc = evaluate_model(model, val_loader, au_feat, weight)
        loss_val.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(root,f'models/ABAW6/{vis}/best_EXPR_{mod_type}_{vi_au}.pth'))
            f1best = f1s
            accbest = acc

        print(f'Validation Loss: {val_loss}, Accuracy: {acc}')
        scheduler.step(val_loss)
    return loss_train, loss_val, best_loss, f1best, accbest

def evaluate_model(model, data_loader, au_feat, weight):
    model.eval()
    total_loss = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        iterator = iter(data_loader)
        while True:
            try:
                EXPR = next(iterator)
                if au_feat == 'nope':
                    vis_feat, y = EXPR[visual_feat], EXPR['label']
                    vis_feat, y = vis_feat.to(device), y.to(device)
                    aud_feat = None
                else:
                    vis_feat, aud_feat, y = EXPR[visual_feat], EXPR[au_feat], EXPR['label']
                    vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
                y_onehot = one_hot_transfer(y, 8).to(device)
                pred, exp_pred = model(vis_feat, aud_feat)
                loss = compute_EXP_loss(pred, y_onehot, weight)
                total_loss.append(loss.item())
                all_preds.extend(exp_pred.cpu().tolist())
                all_targets.extend(y_onehot.cpu().tolist())
            except:
                break

    f1_scores, acc = compute_EXP_F1(all_preds, all_targets)
    return round(np.mean(total_loss),3), round(f1_scores,3), round(acc,3)

train(EXP_model, model_type[0], train_loader, val_loader, 10, 32, optimizer, auft, weights, viau)

# Testing
print('EXP_model')
print(visual_feat + ' & ' + auft)
EXP_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_EXPR_fusion_{viau}.pth'))
EXP_model = EXP_fusion().to(device)
EXP_model.load_state_dict(EXP_best_model)
test_loss, f1s, acc = evaluate_model(EXP_model, test_loader, auft, weights)
print(f'Test set: f1_score {round(f1s,3)}, accuracy: {round(acc,3)}')

print('MLP_model')
print(visual_feat + ' & ' + auft)
mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_EXPR_mlp_{viau}.pth'))
mlp_model = MLPModel().to(device)
mlp_model.load_state_dict(mlp_best_model, strict=False)
val_loss, f1s, acc = evaluate_model(mlp_model, test_loader, auft, weights)
print(f'Test set: f1_score {round(f1s,3)}, accuracy: {round(acc,3)}')

# AU Detection Challenge

AU_model = AU_fusion().to(device)
mlp_model = MLPModel().to(device)

weights = torch.tensor([0.54733899, 0.44180561, 0.56990565, 0.61997328, 0.73956417,0.74692377, 0.72684634, 0.33222808, 0.17383676, 0.20608964, 0.83688068, 0.33890931]).to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, AU_model.parameters()), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, mlp_model.parameters()), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

# Training
def train(model, mod_type, train_loader, val_loader, epoch, batch_size, optim, au_feat, weight, vi_au):
    model.train(True)
    model.eval()
    best_loss = float('inf')
    f1s_best, accbest = 0, 0
    loss_value = []
    loss_train = []
    loss_val = []
    all_preds = []
    all_targets = []

    for e in range(epoch):
        print(f'Epoch: {e+1}')
        torch.manual_seed(2809)
        iterator = iter(train_loader)
        for i in range(len(train_loader)//32):
            try:
                AU = next(iterator)
                if au_feat == 'nope':
                    vis_feat, y = AU[visual_feat], AU['label']
                    vis_feat, y = vis_feat.to(device), y.to(device)
                    aud_feat = None
                else:
                    vis_feat, aud_feat, y = AU[visual_feat], AU[au_feat], AU['label']
                    vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
                model.zero_grad()
                pred, au_pred = model(vis_feat, aud_feat)
                loss = compute_AU_loss(pred, y, weight)
                loss.backward()
                optim.step()
                loss_value.append(loss.item())
                all_preds.extend(au_pred.cpu().tolist())
                all_targets.extend(y.cpu().tolist())
            except:
                break
        avg_loss = round(np.mean(loss_value),3)
        loss_train.append(avg_loss)
        f1_scores, f1_thresh, accuracy, threshold = compute_AU_F1(all_preds, all_targets)
        print(f'Train Loss: {avg_loss}, Accuracy of 12 AU classes: {accuracy}')

        val_loss, f1s, f1t, acc, f1_threshold = evaluate_model(model, val_loader, au_feat, weight)
        loss_val.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(root,f'models/ABAW6/{vis}/best_AU_{mod_type}_{vi_au}.pth'))
            f1best = f1s
            accbest = acc

        print(f'Validation Loss: {val_loss}, Accuracy of 12 AU classes: {acc}')
        scheduler.step(val_loss)
    return loss_train, loss_val, best_loss, f1s_best, f1t_best, accbest

def evaluate_model(model, data_loader, au_feat, weight):
    model.eval()
    total_loss = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        iterator = iter(data_loader)
        for i in range(len(data_loader)//32):
          try:
            AU = next(iterator)
            if au_feat == 'nope':
                vis_feat, y = AU[visual_feat], AU['label']
                vis_feat, y = vis_feat.to(device), y.to(device)
                aud_feat = None
            else:
                vis_feat, aud_feat, y = AU[visual_feat], AU[au_feat], AU['label']
                vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
            pred, au_pred = model(vis_feat, aud_feat)
            loss = compute_AU_loss(pred, y, weight)
            total_loss.append(loss.item())
            all_preds.extend(au_pred.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
          except:
            break

    f1_scores, f1_thresh, acc, threshold = compute_AU_F1(all_preds, all_targets)
    return round(np.mean(total_loss),3), round(f1_scores,3), round(f1_thresh,3), acc, threshold

train(AU_model, model_type[0], train_loader, val_loader, 10, 32, optimizer, auft, weights, viau)

# Testing
print('AU_model')
print(visual_feat + ' & ' + auft)
AU_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_AU_model/best_AU_fusion_{viau}.pth'))
AU_model = AU_fusion().to(device)
AU_model.load_state_dict(AU_best_model)
test_loss, f1s, f1t, acc, threshold = evaluate_model(AU_model, test_loader, auft, weights)
print(f'Test set: f1_score: {f1s}, f1_threshold: {threshold}, accuracy: {acc}')

print('MLP_model')
print(visual_feat + ' & ' + auft)
mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_AU_mlp_{viau}.pth'))
mlp_model = MLPModel(num_classes = 12).to(device)
mlp_model.load_state_dict(mlp_best_model)
test_loss, f1s, f1t, acc, threshold = evaluate_model(mlp_model, test_loader, auft, weights)
print(f'Test set: f1_score: {f1s}, f1_threshold: {threshold}, accuracy: {acc}')

# VA Estimation Challenge
VA_model = VA_fusion().to(device)
mlp_model = MLPModel().to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, VA_model.parameters()), lr=0.00005, betas=(0.9, 0.999), weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, mlp_model.parameters()), lr=0.00005, betas=(0.9, 0.999), weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

# Training
def train(model, mod_type, train_loader, val_loader, epoch, batch_size, optim, scheduler, au_feat, vis_aud):

    model.train(True)
    model.eval()
    best_loss, best_mse = float('inf'), float('inf')
    loss_value = []
    loss_train = []
    loss_val = []
    loss_mse = []
    cc1best, cc2best = 0, 0

    for e in range(epoch):
        print(f'Training Epoch: {e+1}')
        torch.manual_seed(2809)
        iterator = iter(train_loader)
        for i in range(len(train_loader)//32):
            try:
                VA = next(iterator)
                if au_feat == 'nope':
                    vis_feat, y = VA[visual_feat], VA['label']
                    vis_feat, y = vis_feat.to(device), y.to(device)
                    aud_feat = None
                else:
                    vis_feat, aud_feat, y = VA[visual_feat], VA[au_feat], VA['label']
                    vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
                model.zero_grad()
                Vpred, Apred, v_pred, a_pred = model(vis_feat, aud_feat)
                mse_loss, ccc_loss = compute_VA_loss(Vpred, Apred, y)
                ccc_loss.backward()
                optim.step()
                loss_value.append(ccc_loss.item())
                loss_mse.append(mse_loss.item())
                preds = torch.cat((v_pred, a_pred), dim=1)
            except:
                break

        avg_loss = round(np.mean(loss_value),3)
        loss_train.append(avg_loss)
        print(f'Train Loss: {avg_loss}, mse: {round(np.mean(loss_mse),3)}')

        val_loss, mse, ccc1, ccc2 = evaluate_model(model, val_loader, au_feat)
        loss_val.append(val_loss)

        if ccc1 > cc1best:
            torch.save(model.state_dict(), os.path.join(root,f'models/ABAW6/{vis}/best_VA_{mod_type}_{vis_aud}.pth'))
            cc1best = ccc1
            cc2best = ccc2
            best_mse = mse
            best_loss = val_loss

        print(f'Validation Loss: {val_loss}, mse: {mse}')

        scheduler.step(val_loss)
    return loss_train, loss_val, best_loss, best_mse, cc1best, cc2best

def evaluate_model(model, data_loader, au_feat):
    model.eval()
    total_loss = []
    all_targets = []
    all_preds = []
    mse = []
    with torch.no_grad():
        iterator = iter(data_loader)
        for i in range(len(data_loader)//32):
            VA = next(iterator)
            if au_feat == 'nope':
                vis_feat, y = VA[visual_feat], VA['label']
                vis_feat, y = vis_feat.to(device), y.to(device)
                aud_feat = None
            else:
                vis_feat, aud_feat, y = VA[visual_feat], VA[au_feat], VA['label']
                vis_feat, aud_feat, y = vis_feat.to(device), aud_feat.to(device), y.to(device)
            Vpred, Apred, v_pred, a_pred = model(vis_feat, aud_feat)
            mse_loss, ccc_loss = compute_VA_loss(Vpred, Apred, y)
            total_loss.append(ccc_loss.item())
            mse.append(mse_loss.item())
            preds = torch.cat((v_pred, a_pred), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    ccc1, ccc2 = compute_VA_CCC(all_preds, all_targets)
    return round(np.mean(total_loss),3), round(np.mean(mse),3), round(ccc1,3), round(ccc2,3)

train(VA_model, model_type[0], train_loader, val_loader, 10, 32, optimizer, scheduler, auft, viau)

# Testing
print('VA_model')
print(visual_feat + ' & ' + auft)
VA_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_VA_fusion_{viau}.pth'))
VA_model = VA_fusion().to(device)
VA_model.load_state_dict(VA_best_model)
test_loss, mse, ccc1, ccc2 = evaluate_model(VA_model, test_loader, auft)
print(f'Test set: CCC_Valence {ccc1}, CCC_Arousal: {ccc2}')

print('MLP_model')
print(visual_feat + ' & ' + auft)
mlp_best_model = torch.load(os.path.join(root,f'models/ABAW6/{vis}/best_VA_mlp_{viau}.pth'))
mlp_model = MLPModel().to(device)
mlp_model.load_state_dict(mlp_best_model)
test_loss, mse, ccc1, ccc2 = evaluate_model(mlp_model, test_loader, auft)
print(f'Test set: CCC_Valence {ccc1}, CCC_Arousal: {ccc2}')