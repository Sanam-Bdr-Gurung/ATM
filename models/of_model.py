# models/of_model.py
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3,3), p=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class OFEncoder(nn.Module):
    """
    Onsets & Frames-style encoder:
    mel-spectrogram [B, 1, F, T] -> conv2d -> reshape -> BiGRU
    """
    def __init__(self, n_mels=229, hidden=128, gru_layers=2):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d((2,2))  # reduces both F & T dims
        self.conv3 = ConvBlock(64, 96)
        self.conv4 = ConvBlock(96, 128)

        self.gru = nn.GRU(
            input_size=(n_mels//4)*128,  # after two 2x pools in F
            hidden_size=hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True
        )

    def forward(self, x):
        """
        x: [B, 1, F, T] mel-spectrogram (log)
        returns: [B, T', 2*hidden] temporal embeddings
        """
        x = self.conv1(x)       # [B,32,F,T]
        x = self.pool(x)        # [B,32,F/2,T/2]
        x = self.conv2(x)       # [B,64,F/2,T/2]
        x = self.pool(x)        # [B,64,F/4,T/4]
        x = self.conv3(x)       # [B,96,F/4,T/4]
        x = self.conv4(x)       # [B,128,F/4,T/4]
        B, C, Fm, Tm = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,Tm,C,Fm]
        x = x.view(B, Tm, C*Fm)                 # [B,Tm,C*Fm]
        out, _ = self.gru(x)                    # [B,Tm,2H]
        return out

class OFHeads(nn.Module):
    """
    Onset and Frame heads: output per-time pitch logits (MIDI classes).
    """
    def __init__(self, hidden=128, n_pitches=88):
        super().__init__()
        feat = 2*hidden
        self.onset = nn.Linear(feat, n_pitches)
        self.frame = nn.Linear(feat, n_pitches)

    def forward(self, x):
        # x: [B,T,2H]
        onset_logits = self.onset(x)  # [B,T,P]
        frame_logits = self.frame(x)  # [B,T,P]
        return onset_logits, frame_logits

class OnsetsAndFrames(nn.Module):
    """
    Full model wrapper (encoder + heads).
    """
    def __init__(self, n_mels=229, hidden=128, gru_layers=2, n_pitches=88):
        super().__init__()
        self.encoder = OFEncoder(n_mels=n_mels, hidden=hidden, gru_layers=gru_layers)
        self.heads = OFHeads(hidden=hidden, n_pitches=n_pitches)

    def forward(self, mel_log):  # mel_log: [B,1,F,T]
        feats = self.encoder(mel_log)
        onset_logits, frame_logits = self.heads(feats)
        return onset_logits, frame_logits
