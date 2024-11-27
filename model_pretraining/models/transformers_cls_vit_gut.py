import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

def length_to_mask(lengths, max_len = None):
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length + 1, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]

def length_to_mask_concate(lengths, max_len = None):
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).to('cuda')
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sin and cos position encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dim, 
                 dim_feedforward, hidden_dim_fc, 
                 dim_feedforward_concate, hidden_dim_fc_concate,
                 num_class = 2, num_head=1, num_layers=1, dropout=0,
                 max_len=3000, activation: str = "relu"):
        super(Transformer, self).__init__()
        self.region_size = [[24, 24, 16], [24, 24, 16], [24, 24, 16], [24, 24, 16], [24, 24, 16], [24, 24, 16], [24, 24, 16], 
                            [24, 24, 16], [24, 24, 16], [24, 24, 16], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], 
                            [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], 
                            [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], [32, 32, 24], 
                            [32, 32, 24], [32, 32, 24], [24, 24, 24], [24, 24, 24], [24, 24, 24], [24, 24, 24], [24, 24, 24], 
                            [24, 24, 24], [24, 24, 24], [24, 24, 24], [24, 24, 24], [24, 24, 24], [30, 30, 8], [30, 30, 8], 
                            [30, 30, 8], [30, 30, 8], [30, 30, 8], [30, 30, 8], [30, 30, 8], [30, 30, 8], [30, 30, 8], [30, 30, 8]]
        self.hidden_dim_fc = hidden_dim_fc
        self.patch_size = 8

        self.projection_gut = nn.Linear(embedding_dim, embedding_dim)
        self.position_embedding_gut = PositionalEncoding(embedding_dim, dropout, max_len)
        encoder_layer_gut = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer_gut = nn.TransformerEncoder(encoder_layer_gut, num_layers)

        self.conv3d = nn.Conv3d(1, 1, kernel_size = 9, stride = 1, padding = 4)


    def merge_patches(self, patches, original_shape):
        patch_size = 8
        _, b, _ = patches.size()
        unfolded_shape = [(s // patch_size) for s in original_shape]
        reconstructed_image = patches.view(unfolded_shape[0], unfolded_shape[1], unfolded_shape[2], b, patch_size, patch_size, patch_size)
        reconstructed_image = reconstructed_image.permute(3, 0, 4, 1, 5, 2, 6).contiguous().view(b, original_shape[0], original_shape[1], original_shape[2])

        return reconstructed_image


    def vit(self, inputs, lengths):
        b, _, _ = inputs.shape
        inputs = torch.transpose(inputs, 0, 1)

        hidden_states = self.position_embedding_gut(inputs)
        hidden_states = self.transformer_gut(hidden_states.float())
        return hidden_states


    def forward(self, inputs_4D, lengths_4D):
        region_num, b, _, dim = inputs_4D.shape

        imgrecons = []
        for idx in range(region_num):
            inputs = (inputs_4D[idx, :, :, :])

            lengths = (lengths_4D[idx, :])
            current_length = lengths[0].item() if lengths.dim() > 0 else lengths.item()
            inputs = inputs[:, :current_length, :]
            inputs = inputs.reshape(-1, dim)
            inputs = self.projection_gut(inputs.float())
            inputs = inputs.reshape(b, -1, dim)

            hidden_states = self.vit(inputs, lengths)
            imgrecon = self.merge_patches(hidden_states, self.region_size[idx])
            imgrecon = imgrecon.unsqueeze(1)
            imgrecon = self.conv3d(imgrecon)
            imgrecon = imgrecon.squeeze()
            imgrecons.append(imgrecon)
            del imgrecon
        return imgrecons
