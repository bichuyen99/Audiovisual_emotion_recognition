import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import warnings
import torchaudio
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = '/MSc/Thesis'

# Convert video to audio
def vid2aud(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in tqdm(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, file_name)
        audio_file_name = os.path.splitext(file_name)[0] + '.wav'
        audio_file_path = os.path.join(output_folder, audio_file_name)
        if os.path.exists(audio_file_path):
            continue
        if os.path.isfile(video_path) and file_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file_path)
            audio_clip.close()
            video_clip.close()

for i in ['batch1','batch2','new_vids']:
    video_folder= os.path.join(root,'data/video', i)
    print(f'\nProcessing {video_folder}')
    output_folder= os.path.join(root, 'data/audio')
    vid2aud(video_folder, output_folder)

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

# Visual feature

IMG_SIZE=224
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)
test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

with open('/content/drive/MyDrive/MSc/Thesis/data/test_list.txt', 'r') as f:
      test_list = f.read().splitlines()

def extract_visual_feature(mode_name, typ):
    print('loading model:',mode_name)
    feature_extractor_model = torch.load(os.path.join(root, f'models/EmotiEffNet/enet/{mode_name}.pt'))
    feature_extractor_model.classifier=torch.nn.Identity()
    feature_extractor_model=feature_extractor_model.to(device)
    feature_extractor_model.eval()

    if typ == 'cropped_aligned':
        dir = ['cropped_aligned','cropped_aligned_new_50_vids']
    elif typ == 'cropped':
        dir = ['batch1', 'batch2', 'cropped_new_50_vids']

    for d in dir:
        root_vis = f'/content/data/{d}'
        print(f'processing {root_vis}')
        save_folder = os.path.join(root, f'models/ABAW6/visualfeat_{mode_name}_{typ}')
        os.makedirs(save_folder, exist_ok=True)
        for filename in tqdm(os.listdir(root_vis)):
            X_features=[]
            img_names=[]
            img_feat = {}
            imgs=[]
            frames_dir=os.path.join(root_vis,filename)

            if not os.path.isdir(frames_dir):
                continue
            save_file = os.path.join(save_folder, filename+'.npy')
            if os.path.exists(save_file):
                continue
            else:
                for img_name in os.listdir(frames_dir):
                    if img_name.lower().endswith('.jpg'):
                        img = Image.open(os.path.join(frames_dir,img_name))
                        if filename in test_list:
                            img_tensor = test_transforms(img)
                        else:
                            img_tensor = train_transforms(img)
                        if img.size:
                            img_names.append(filename+'/'+img_name)
                            imgs.append(img_tensor)
                            if len(imgs)>= 64:
                                features = feature_extractor_model(torch.stack(imgs, dim=0).to(device))
                                features = features.data.cpu().numpy()
                                if len(X_features)==0:
                                    X_features=features
                                else:
                                    X_features=np.concatenate((X_features,features),axis=0)
                                imgs=[]

                if len(imgs)>0:
                    features = feature_extractor_model(torch.stack(imgs, dim=0).to(device))
                    features = features.data.cpu().numpy()

                    if len(X_features)==0:
                        X_features=features
                    else:
                        X_features=np.concatenate((X_features,features),axis=0)

                    imgs=[]
                img_feat= {img_name:global_features for img_name,global_features in zip(img_names,X_features)}
                np.save(save_file,img_feat)

# Extract visual feature from cropped images by using EfficientNet_B2
extract_visual_feature('enet_b2_8_best','cropped')

# Extract visual feature from cropped_aligned images by using EfficientNet_B2
extract_visual_feature('enet_b2_8_best','cropped_aligned')

# Audio feature
video2len={}
for i in ['batch1', 'batch2', 'new_vids']:
    d = os.path.join('/content/video',i)
    for filename in os.listdir(d):
        fn, ext = os.path.splitext(os.path.basename(filename))
        vid=os.path.join(d,filename)
        cap = cv2.VideoCapture(vid)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video2len[fn]=total_frames+1

for filename in os.listdir(os.path.join(root,f'models/ABAW6/visualfeat_enet_b2_8_best_cropped_aligned')):
    if 'left' in filename or 'right' in filename:
        feature_path = os.path.join(root, filename)
        feature = np.load(feature_path, allow_pickle=True).tolist()
        fn = filename.split('.')[0]
        video2len[fn] = len(feature)

with open('/content/drive/MyDrive/MSc/Thesis/data/vid_length.pkl', 'wb') as f:
    pickle.dump(video2len, f)

def extract_audio_feature(mode_name):

    if mode_name == 'wav2vec2':
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model().to(device)
    elif mode_name == 'vggish':
        model = torch.hub.load('harritaylor/torchvggish', mode_name)
        model.eval().to(device)

    print('loading model:',mode_name)
    root_wav = os.path.join(root, 'data/audio')
    wav_files = os.listdir(root_wav)[::-1]
    print(f'processing {root_wav}')
    save_folder = os.path.join(root, f'models/ABAW6/audiofeat_{mode_name}')
    os.makedirs(save_folder, exist_ok=True)

    with open('/content/drive/MyDrive/MSc/Thesis/data/vid_length.pkl', 'rb') as f:
        video2len = pickle.load(f)

    for nwav, frames_count in tqdm(video2len.items()):
        if nwav.endswith('_left'):
            wav_f = nwav[:-5]
        elif nwav.endswith('_right'):
            wav_f = nwav[:-6]
        else:
            wav_f = nwav
        audio_features = {}
        save_file = os.path.join(save_folder, nwav +'.npy')
        if os.path.exists(save_file):
            continue
        if mode_name == 'vggish':
            with torch.no_grad():
                reps = model.forward(os.path.join(root_wav, wav_f + '.wav'))
                reps = reps.cpu().numpy()/255.
        else:
            wav, rate = torchaudio.load(os.path.join(root_wav, wav_f + '.wav'))
            if rate!= bundle.sample_rate:
                wav = torchaudio.functional.resample(wav, rate, bundle.sample_rate)
            reps = []
            channel, length = wav.shape
            max_length = 2500000
            with torch.no_grad():
                for i in range(length//max_length+1):
                    reps.append(model.extract_features(wav.cuda()[:,i*max_length:(i+1)*max_length])[0][-1])
            reps = torch.concatenate(reps,dim=1)
            if channel !=1:
                reps = torch.mean(reps, dim=0).unsqueeze(dim=0)

            reps = reps.cpu().numpy().squeeze(0)
        audio_scale=len(reps)/frames_count

        for frame_number in range(frames_count):
            ind=int(frame_number*audio_scale)
            nframe = get_names(nwav,frame_number+1)
            audio_features[nframe]= reps[ind]

        np.save(save_file, audio_features)

# Extract audio feature by using wav2vec2
extract_audio_feature('wav2vec2')

# Extract audio feature by using vggish
extract_audio_feature('vggish')