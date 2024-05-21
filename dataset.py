import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import random
import warnings
warnings.filterwarnings("ignore")

root = '/MSc/Thesis'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

# Creates a dataset for loading multimodal (visual and audio) features
# with corresponding labels for the ABAW challenge tasks
class ABAW_dataset(Dataset):
    def __init__(self, root, split, typ, task, feature_v, feature_a):
        self.root = root
        self.split = split
        self.typ = typ
        self.task = task
        self.anno_path = os.path.join(self.root,f'data/Annotations/{self.task}/{self.split}')
        self.feature_v, self.feature_a = feature_v, feature_a
        self.feature = [self.feature_v, self.feature_a]
        with open(os.path.join(root, f'data/Annotations/{self.task}/{self.typ}.txt'), 'r') as f:
                self.vidnames = f.read().splitlines()
        self.feature_dims = 0
        self.data = {}
        self.data[self.task] = {}
        self.iname = []
        for feature_name in self.feature:
            if 'visual' in feature_name:
                self.data = self.load_feature_v(feature_name)
            elif 'audio' in feature_name:
                self.data = self.load_feature_a(feature_name)

    def get_names(vid, id):
            name = ""
            if id>=0 and id<10:
                name = f"{vid}/0000" + str(id) + ".jpg"
            elif id>=10 and id<100:
                name = f"{vid}/000" + str(id) + ".jpg"
            elif id>=100 and id<1000:
                name = f"{vid}/00" + str(id) + ".jpg"
            elif id>=1000 and id<10000:
                name = f"{vid}/0" + str(id) + ".jpg"
            else:
                name = f"{vid}/" + str(id) + ".jpg"
            return name

    def load_feature_v(self, feature_v):
            print(f'loading visual feature: {feature_v}')
            feat_root = os.path.join(root + '/models/ABAW6', feature_v)
            filenames = os.listdir(feat_root)[:]
            for vname in tqdm(self.vidnames):
                    feature = np.load(os.path.join(feat_root, f'{vname}.npy'), allow_pickle=True).tolist()
                    with open(os.path.join(self.anno_path, f'{vname}.txt')) as f:
                        labels = f.read().splitlines()
                    self.data[self.task][vname] = {}

                    for imgname, val in feature.items():
                        for i,line in enumerate(labels):
                            if i > 0:
                                imname = get_names(vname, i)
                                if imname == imgname:
                                    if self.task == 'AU_Detection_Challenge':
                                        splitted_line=line.split(',')
                                        aus = list(map(int,splitted_line))
                                        if min(aus) >= 0:
                                            labs = torch.tensor(aus)
                                            self.data[self.task][vname][imgname] = {f'{feature_v}': val, 'label': labs}
                                            self.iname.append(imname)
                                    elif self.task == 'VA_Estimation_Challenge':
                                        splitted_line=line.split(',')
                                        valence=float(splitted_line[0])
                                        arousal=float(splitted_line[1])
                                        if valence >= -1 and arousal >= -1:
                                            labs = torch.tensor([valence, arousal])
                                            self.data[self.task][vname][imgname] = {f'{feature_v}': val, 'label': labs}
                                            self.iname.append(imname)
                                    elif self.task == 'EXPR_Recognition_Challenge':
                                        exp = int(line)
                                        if exp >= 0:
                                            labs = torch.tensor(exp)
                                            self.data[self.task][vname][imgname] = {f'{feature_v}': val, 'label': labs}
                                            self.iname.append(imname)
                    self.feature_dims += len(self.data[self.task][vname])
            return self.data

    def load_feature_a(self, feature_a):
            print(f'loading audio feature: {feature_a}')
            feat_root = os.path.join(root + '/models/ABAW6', feature_a)
            filenames = os.listdir(feat_root)[:]
            for vname in tqdm(self.vidnames):
                    feature = np.load(os.path.join(feat_root, f'{vname}.npy'), allow_pickle=True).tolist()
                    for imgname, val in feature.items():
                        if imgname in self.data[self.task][vname]:
                            self.data[self.task][vname][imgname].update({f'{feature_a}': val})

                    for img, value in list(self.data[self.task][vname].items()):
                        if len(value) < 3:
                            self.data[self.task][vname].pop(img)
            return self.data

    def __getitem__(self, index):
            frame = self.iname[index]
            vname = frame.split('/')[0]
            data = self.data[self.task][vname][frame]
            data['frame'] = frame
            data['vid'] = vname
            data['label'] = self.data[self.task][vname][frame]['label']
            return data

    def __len__(self):
            return self.feature_dims

class ABAW_dataset1(Dataset):
    def __init__(self, data, iname, dims, task):
        self.data = data
        self.iname = iname
        self.task = task
        self.feature_dims = dims
    def __getitem__(self, index):
        frame = self.iname[index]
        vname = frame.split('/')[0]
        data = self.data[self.task][vname][frame]
        data['frame'] = frame
        data['vid'] = vname
        data['label'] = self.data[self.task][vname][frame]['label']
        return data

    def __len__(self):
            return self.feature_dims

# Generate text files listing annotation filenames for each challenge
for d in ['VA_Estimation_Challenge','EXPR_Recognition_Challenge','AU_Detection_Challenge']:
    data_dir=os.path.join(root,'data/Annotations',d)
    for k in ['Train_Set','Validation_Set']:
        data_label=os.path.join(data_dir,k)
        with open(os.path.join(data_dir,f'{k}.txt'), 'w') as f:
            for filename in tqdm(os.listdir(data_label)):
                fn, ext = os.path.splitext(os.path.basename(filename))
                if ext.lower()=='.txt':
                    f.write(fn+'\n')

# Split data for each challenge into train/validation sets from the 'Train_Set.txt' file
# and test sets from the 'Validation_Set.txt' file.
test_list = []

for d in ['VA_Estimation_Challenge','EXPR_Recognition_Challenge','AU_Detection_Challenge']:
    print(d)
    with open(os.path.join(root,f'data/Annotations/{d}/Train_Set.txt'), 'r') as f:
        files = f.read().splitlines()
    random.shuffle(files)
    ratio = int(len(files)/5)
    val_set = files[:ratio]
    test_list.extend(val_set)
    train_set = files[ratio:]
    print('Train_set:')
    with open(os.path.join(root,f'data/Annotations/{d}/Train.txt'), 'w') as f:
        for ftrain in tqdm(train_set):
            f.write(ftrain+'\n')
    print('Val_set:')
    with open(os.path.join(root,f'data/Annotations/{d}/Val.txt'), 'w') as f:
        for fval in tqdm(val_set):
            f.write(fval+'\n')

    with open(os.path.join(root,f'data/Annotations/{d}/Validation_Set.txt'), 'r') as f:
        test_set = f.read().splitlines()
    test_list.extend(test_set)
    print('Test_set:')
    with open(os.path.join(root,f'data/Annotations/{d}/Test.txt'), 'w') as f:
        for ftest in tqdm(test_set):
            f.write(ftest+'\n')
with open(os.path.join(root,'data/test_list.txt'), 'w') as f:
    for fn in test_list:
        f.write(fn+'\n')

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

# Example of loading dataset from EXPR_Recognition_Challenge
train_set = ABAW_dataset(root, split[0], typ[0], task[0], feature_v=visual_feat, feature_a=audio_feat)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

val_set = ABAW_dataset(root, split[0], typ[1], task[0], feature_v=visual_feat, feature_a=audio_feat)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

test_set = ABAW_dataset(root, split[1], typ[2], task[0], feature_v=visual_feat, feature_a=audio_feat)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)