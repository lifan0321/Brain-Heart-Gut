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
        self.region_num_brain = 91 #91+23+50
        self.hidden_dim_fc = hidden_dim_fc

        self.projection_brain = nn.Linear(embedding_dim, embedding_dim)

        self.cls_token_brain = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding_brain = PositionalEncoding(embedding_dim, dropout, max_len)
        encoder_layer_brain = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer_brain = nn.TransformerEncoder(encoder_layer_brain, num_layers)

        self.fc_layer_1_brain = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim_fc) for _ in range(self.region_num_brain)])
        self.fc_layer_2_brain = nn.ModuleList([nn.Linear(hidden_dim_fc, num_class) for _ in range(self.region_num_brain)])

        self.cls_token_concate_brain = nn.Parameter(torch.randn(1, 1, hidden_dim_fc))
        self.position_embedding_concate_brain = PositionalEncoding(hidden_dim_fc, dropout, max_len)
        encoder_layer_concate_brain = nn.TransformerEncoderLayer(hidden_dim_fc, num_head, dim_feedforward_concate, dropout, activation)
        self.transformer_concate_brain = nn.TransformerEncoder(encoder_layer_concate_brain, num_layers)

        self.fc_layer_1_concate_brain = nn.Linear(hidden_dim_fc, hidden_dim_fc_concate)
        self.fc_layer_2_concate_brain = nn.Linear(hidden_dim_fc_concate, num_class)

    def vit(self, inputs, lengths):
        b, _, _ = inputs.shape
        cls_tokens = repeat(self.cls_token_brain, '() n e -> b n e', b = b)
        inputs = torch.cat([cls_tokens, inputs], dim = 1)
        inputs = torch.transpose(inputs, 0, 1)

        hidden_states = self.position_embedding_brain(inputs)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer_brain(hidden_states.float())
        hidden_states = hidden_states[0, :, :]
        return hidden_states

    def vit_concate(self, inputs, lengths):
        b, _, _ = inputs.shape
        inputs = torch.transpose(inputs, 0, 1)

        hidden_states = self.position_embedding_concate_brain(inputs)
        attention_mask = length_to_mask_concate(lengths) == False
        hidden_states = self.transformer_concate_brain(hidden_states.float())

        hidden_states = hidden_states.mean(dim=0)
        return hidden_states

    def forward(self, inputs_4D, lengths_4D):
        region_num_brain, b, _, dim = inputs_4D.shape
        region_fc = []
        log_probs_region_all = []
        for idx in range(region_num_brain):
            inputs = (inputs_4D[idx, :, :, :])

            lengths = (lengths_4D[idx, :])
            current_length = lengths[0].item() if lengths.dim() > 0 else lengths.item()
            inputs = inputs[:, :current_length, :]
            inputs = inputs.reshape(-1, dim)
            inputs = self.projection_brain(inputs.float())
            inputs = inputs.reshape(b, -1, dim)

            hidden_states = self.vit(inputs, lengths)
            fc1 = self.fc_layer_1_brain[idx](hidden_states)
            
            region_fc.append(fc1)
            output_region = self.fc_layer_2_brain[idx](fc1)
            log_probs_region = F.softmax(output_region, dim=1)
            log_probs_region_all.append(log_probs_region)

        region_fc = torch.stack(region_fc, dim = 1)
        region_fc = region_fc.view(b, self.region_num_brain, self.hidden_dim_fc)

        lengths_all = self.region_num_brain * torch.ones(b).to('cuda')
        fc1_vit = self.vit_concate(region_fc, lengths_all)
        hidden = self.fc_layer_1_concate_brain(fc1_vit)
        output = self.fc_layer_2_concate_brain(hidden)
        log_probs = F.softmax(output, dim=1)

        log_probs_region_all.append(log_probs)
        return log_probs_region_all, region_fc
