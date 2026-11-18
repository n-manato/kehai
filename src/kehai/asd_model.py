"""LR-ASD (Liao et al., "LR-ASD: Lightweight and Robust Network for Active Speaker
Detection", IJCV 2025) — the network only, CPU inference.

Vendored from github.com/airmeet/LR-ASD (model/Model.py, model/Encoder.py,
model/Classifier.py, loss.py) with training-only parts removed. Weights:
models/asd/finetuning_TalkSet.model (the authors' release, trained on AVA and
fine-tuned on TalkSet = VoxCeleb2 + LRS3 talking heads).

MIT License — Copyright (c) 2025 Liao Junhua (full text: models/asd/LICENSE.LR-ASD)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _AudioBlock(nn.Module):
    def __init__(self, cin, cout, k1, k2):
        super().__init__()
        self.relu = nn.ReLU()
        p1, p2 = (k1 - 1) // 2, (k2 - 1) // 2
        self.m_1 = nn.Conv2d(cin, cout // 2, kernel_size=(k1, 1), padding=(p1, 0), bias=False)
        self.m_norm_1 = nn.BatchNorm2d(cout // 2, momentum=0.01, eps=0.001)
        self.m_2 = nn.Conv2d(cout // 2, cout, kernel_size=(k2, 1), padding=(p2, 0), bias=False)
        self.m_norm_2 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_1 = nn.Conv2d(cout, cout, kernel_size=(1, k1), padding=(0, p1), bias=False)
        self.t_norm_1 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv2d(cout, cout, kernel_size=(1, k2), padding=(0, p2), bias=False)
        self.t_norm_2 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        return self.relu(self.t_norm_2(self.t_2(x)))


class _VisualBlock(nn.Module):
    def __init__(self, cin, cout, k1, k2, is_down=False):
        super().__init__()
        self.relu = nn.ReLU()
        p1, p2 = (k1 - 1) // 2, (k2 - 1) // 2
        stride = (1, 2, 2) if is_down else 1
        self.s_1 = nn.Conv3d(cin, cout // 2, kernel_size=(1, k1, k1), stride=stride, padding=(0, p1, p1), bias=False)
        self.s_norm_1 = nn.BatchNorm3d(cout // 2, momentum=0.01, eps=0.001)
        self.s_2 = nn.Conv3d(cout // 2, cout, kernel_size=(1, k2, k2), padding=(0, p2, p2), bias=False)
        self.s_norm_2 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.t_1 = nn.Conv3d(cout, cout, kernel_size=(k1, 1, 1), padding=(p1, 0, 0), bias=False)
        self.t_norm_1 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv3d(cout, cout, kernel_size=(k2, 1, 1), padding=(p2, 0, 0), bias=False)
        self.t_norm_2 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        return self.relu(self.t_norm_2(self.t_2(x)))


class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _VisualBlock(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128, 5, 3)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x).transpose(1, 2)
        b, t, c, w, h = x.shape
        x = self.maxpool(x.reshape(b * t, c, w, h))
        return x.view(b, t, c)


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _AudioBlock(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = _AudioBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = _AudioBlock(64, 128, 5, 3)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class _Fusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(channel, momentum=0.01, eps=0.001)

    def forward(self, x1, x2):
        identity = torch.cat((x1, x2), 2).transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        return (identity * w).transpose(1, 2)


class _Detector(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, batch_first=True)
        self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, batch_first=True)
        self.attention = _Fusion(channel // 2)

    def forward(self, x):
        x1, _ = self.gru_forward(x)
        x2, _ = self.gru_backward(torch.flip(x, dims=[1]))
        return self.attention(x1, torch.flip(x2, dims=[1]))


class ASDModel(nn.Module):
    """forward(audio (1, 4T, 13) MFCC @100 Hz, video (1, T, 112, 112) gray @25 fps)
    -> per-video-frame logit of "this face is the one speaking" (> 0 = speaking)."""

    def __init__(self):
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.fusion = _Fusion(256)
        self.detector = _Detector(256)
        self.fc = nn.Linear(128, 2)  # lossAV.FC
        # lossV.FC: the picture-only head the authors train beside the
        # audio-visual one ("is this face speaking", from the lips alone).
        self.fc_v = nn.Linear(128, 2)

    def visual(self, video: torch.Tensor) -> torch.Tensor:
        """(1, T, 112, 112) gray 0-255 -> (1, T, 128). The costly part (~47 ms
        per second of video on 4 CPU threads); each frame sees +-9 neighbours."""
        b, t, w, h = video.shape
        return self.visualEncoder((video.view(b, 1, t, w, h) / 255 - 0.4161) / 0.1688)

    def audio(self, mfcc: torch.Tensor) -> torch.Tensor:
        """(1, 4T, 13) -> (1, T, 128)."""
        return self.audioEncoder(mfcc.unsqueeze(1).transpose(2, 3))

    def backend(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-frame logits of "speaking" from aligned audio/visual embeddings."""
        return self.fc(self.detector(self.fusion(a, v)).reshape(-1, 128))[:, 1]

    def visual_logits(self, v: torch.Tensor) -> torch.Tensor:
        """Per-frame logit of "speaking" from the visual embeddings alone."""
        return self.fc_v(v.reshape(-1, 128))[:, 1]

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        return self.backend(self.audio(audio), self.visual(video))


def load(path: str) -> ASDModel:
    m = ASDModel()
    state = torch.load(path, map_location="cpu", weights_only=True)
    own = m.state_dict()
    for name, val in state.items():
        key = name.replace("module.", "")
        if key.startswith("model."):
            key = key[len("model."):]
        elif key.startswith("lossAV.FC."):
            key = "fc." + key[len("lossAV.FC."):]
        elif key.startswith("lossV.FC."):
            key = "fc_v." + key[len("lossV.FC."):]
        else:
            continue
        if key in own and own[key].shape == val.shape:
            own[key].copy_(val)
        else:
            raise KeyError(f"LR-ASD weight {name} does not fit ({key})")
    m.eval()
    return m
