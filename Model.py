import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from collections import OrderedDict
from sparse_transformer import Transformer


class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)  # Shape: [batch_size, seq_len, input_size*2]
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_size*2]
        # Initialize the hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Shape: [1, batch_size, hidden_size]

        # Pass through GRU
        out, _ = self.gru(x, h0)  # out shape: [batch_size, seq_len, hidden_size]

        out = out.squeeze(1)  # Shape: [batch_size, hidden_size]
        out = self.fc(out)  # Shape: [batch_size, hidden_size]

        return out


class MSSTrans(nn.Module):
    def __init__(self, label_dim=1):
        super(MSSTrans, self).__init__()

        src_pad_idx = 0
        src_vocab_size = 256
        self.transformer_4096 = Transformer(src_vocab_size, src_pad_idx)  # num_layers=6, heads=8
        self.transformer_256 = Transformer(src_vocab_size, src_pad_idx)
        self.transformer_16 = Transformer(src_vocab_size, src_pad_idx)

        self.fusion_4096_256 = FusionGRU(input_size=src_vocab_size*2, hidden_size=src_vocab_size)
        self.fusion_256_16 = FusionGRU(input_size=src_vocab_size * 2, hidden_size=src_vocab_size)

        self.classifier = nn.Sequential(nn.Linear(256, label_dim))

    def forward(self, img_4096_, img_256_, img_16_):  # [B, 1, 64, 256], [B, 256, 64, 256], [B, 256*256, 64, 256]
        k = 64
        img_4096, attn_4096 = self.transformer_4096(torch.squeeze(img_4096_))  # [B, 256]
        attn_mean = attn_4096.mean(dim=2, keepdim=True)
        topk_values, topk_indices = torch.topk(attn_mean, k, dim=1)

        img_256, attn_256 = self.transformer_4096(torch.squeeze(img_256_[:,:,topk_indices[1],:]))
        attn_mean = attn_256.mean(dim=2, keepdim=True)
        topk_values, topk_indices = torch.topk(attn_mean, k, dim=1)

        img_16, attn_16 = self.transformer_4096(torch.squeeze(img_16_[:,:,topk_indices[1],:]))

        img_fusion_4096_256 = self.fusion_4096_256(img_4096, img_256)  # [B, 256]
        img_fusion_256_16 = self.fusion_256_16(img_fusion_4096_256, img_16)

        prediction = self.classifier(img_fusion_256_16.view(img_fusion_256_16.shape[0], -1))

        return prediction
