import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
auft = audio_feat[0]

# EXPR Recognition Challenge
# Transformer-based model for EXPR
class EXP_fusion(nn.Module):
    def __init__(self, batchsize = batch_size, audio_ft = auft, hidden_size = [512, 128, batch_size]):
        super(EXP_fusion, self).__init__()
        self.batchsize = batchsize
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.hidden_size = hidden_size
        self.feat_fc = nn.Conv1d(self.concat_dim, hidden_size[0], 1, padding=0)
        self.activ = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv1d(hidden_size[0], hidden_size[1], 1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size[1], nhead=4, dim_feedforward=hidden_size[1], dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[2]),
                nn.BatchNorm1d(hidden_size[2]),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_size[2], 8))

    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs,dim=1)
        feat = torch.transpose(feat,0,1)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.conv1(feat)
        out = torch.transpose(out,0,1)
        out = self.transformer_encoder(out)
        out = self.head(out)

        return out, torch.softmax(out, dim = 1)

# MLP model for EXPR
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
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs, dim=1)
        feat = self.fc1(feat)
        feat = self.activ(feat)
        out = self.fc2(feat)
        return out, torch.softmax(out, dim=1)

# AU Detection Challenge
# Transformer-based model for AU
class AU_fusion(nn.Module):
    def __init__(self, batchsize = batch_size, audio_ft = auft, hidden_size = [512, 128, batch_size]):
        super(AU_fusion, self).__init__()
        self.batchsize = batchsize
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.hidden_size = hidden_size
        self.feat_fc = nn.Conv1d(self.concat_dim, hidden_size[0], 1, padding=0)
        self.activ = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv1d(hidden_size[0], hidden_size[1], 1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size[1], nhead=4, dim_feedforward=hidden_size[1], dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[2]),
                nn.BatchNorm1d(hidden_size[2]),
                nn.Linear(hidden_size[2], 12))

    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs,dim=1)
        feat = torch.transpose(feat,0,1)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.conv1(feat)
        out = torch.transpose(out,0,1)
        out = self.transformer_encoder(out)
        out = self.head(out)

        return out, torch.sigmoid(out)

# MLP model for AU
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
            feat = torch.cat(inputs, dim=1)
        feat = self.fc1(feat)
        feat = self.activ(feat)
        out = self.fc2(feat)
        return out, torch.sigmoid(out)

# VA Estimation Challenge
# Transformer-based model for VA
class VA_fusion(nn.Module):
    def __init__(self, batchsize = batch_size, audio_ft = auft, hidden_size = [512, 128, batch_size]):
        super(VA_fusion, self).__init__()
        self.batchsize = batchsize
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.hidden_size = hidden_size
        self.feat_fc = nn.Conv1d(self.concat_dim, hidden_size[0], 1, padding=0)
        self.activ = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv1d(hidden_size[0], hidden_size[1], 1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size[1], nhead=4, dim_feedforward=hidden_size[1], dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.vhead = nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[2]),
                nn.BatchNorm1d(hidden_size[2]),
                nn.Linear(hidden_size[2], 1),
                )
        self.ahead = nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[2]),
                nn.BatchNorm1d(hidden_size[2]),
                nn.Linear(hidden_size[2], 1),
                )

    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs,dim=1)
        feat = torch.transpose(feat,0,1)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.conv1(feat)
        out = torch.transpose(out,0,1)
        out = self.transformer_encoder(out)
        vout = self.vhead(out)
        aout = self.ahead(out)

        return vout, aout, torch.tanh(vout), torch.tanh(aout)

# MLP model for VA
class MLPModel(nn.Module):
    def __init__(self, audio_ft = auft, num_classes=1):
        super(MLPModel, self).__init__()
        if audio_ft == 'audiofeat_wav2vec2':
            self.concat_dim = 2176    #1408+768
        elif audio_ft == 'audiofeat_vggish':
            self.concat_dim = 1536    #1408+128
        elif audio_ft == 'nope':
            self.concat_dim = 1408    #visual only
        self.feat_fc = nn.Conv1d(self.concat_dim, 512, 1, padding=0)
        self.vhead = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

        self.ahead = nn.Sequential(
            nn.Linear(self.concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    def forward(self, vis_feat, aud_feat):
        if aud_feat == None:
            feat = vis_feat
        else:
            inputs = [vis_feat]
            inputs.append(aud_feat)
            feat = torch.cat(inputs, dim=1)
        vfeat = self.feat_fc(torch.transpose(feat,0,1))
        vfeat = torch.transpose(vfeat,0,1)
        vout = self.vhead(vfeat)
        aout = self.ahead(feat)

        return vout, aout, torch.tanh(vout), torch.tanh(aout)
