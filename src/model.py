import torch.nn as nn
import torch
import math
import numpy as np
from torch.nn.modules.normalization import LayerNorm
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch.nn.functional as F


def weight_init_normal(weight, normal_std):
    nn.init.normal_(weight, 0.0, normal_std)


def bias_init(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)


class PositionalEncoding(nn.Module):
    """
    For positional encoding in transformer.

    """

    def __init__(self, d_model, pos_enc_start=0, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(pos_enc_start, max_len,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, emb_scale=0.5, pad_idx=0):
        super(TokenEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj**emb_scale

        self.emb_lookup = nn.Embedding(n_token, d_embed, padding_idx=pad_idx)
        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)

        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)


class Generator(nn.Module):

    def __init__(
        self,
        ntoken,
    ):
        super(Generator, self).__init__()

        self.n_token = ntoken
        self.n_layer = 12
        self.n_head = 8
        self.d_model = 512
        self.d_ff = 2048
        self.dropout = 0.1
        self.d_embed = 512

        gpt_config = GPT2Config(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.d_model,
            n_inner=self.d_ff,
            resid_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            max_position_embeddings=4096,
        )
        self.transformer_decoder = nn.ModuleList(
            [GPT2Block(gpt_config, layer_idx=i) for i in range(self.n_layer)])

        # positional encoding used for encoder
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # token embedding
        self.token_embedding = TokenEmbedding(self.n_token, self.d_embed,
                                              self.d_model)

        # output layer
        self.output_layer = nn.Linear(self.d_model, self.n_token)

        self.apply(weights_init)

    def forward(self, tgt):

        tgt = self.token_embedding(tgt)
        tgt = self.pos_encoding(tgt)

        for i in range(self.n_layer):
            tgt = self.transformer_decoder[i].forward(tgt)[0]

        out = self.output_layer(tgt)
        return out

    def temperature(self, logits, t):
        if np.isinf(np.exp(logits / t)).any() or np.isinf(
                np.sum(np.exp(logits / t))):
            probs = np.zeros(logits.shape)
            probs[np.argmax(logits)] = 1
        else:
            probs = np.exp(logits / t) / np.sum(np.exp(logits / t))
        return probs

    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word


class ChordModel(nn.Module):

    def __init__(
        self,
        ntoken,
    ):
        super(ChordModel, self).__init__()

        self.n_token = ntoken
        self.n_layer = 6
        self.n_head = 8
        self.d_model = 256
        self.d_ff = 2048
        self.dropout = 0.1
        self.d_embed = 256

        gpt_config = GPT2Config(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.d_model,
            n_inner=self.d_ff,
            resid_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            max_position_embeddings=4096,
        )
        self.transformer_decoder = nn.ModuleList(
            [GPT2Block(gpt_config, layer_idx=i) for i in range(self.n_layer)])

        # positional encoding used for encoder
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # token embedding
        self.token_embedding = TokenEmbedding(self.n_token, self.d_embed,
                                              self.d_model)

        # output layer
        self.output_layer = nn.Linear(self.d_model, self.n_token)

        self.apply(weights_init)

    def forward(self, tgt):

        tgt = self.token_embedding(tgt)
        tgt = self.pos_encoding(tgt)

        for i in range(self.n_layer):
            tgt = self.transformer_decoder[i].forward(tgt)[0]

        out = self.output_layer(tgt)
        return out


class MelodyModel(nn.Module):

    def __init__(
        self,
        ntoken,
    ):
        super(MelodyModel, self).__init__()

        self.n_token = ntoken
        self.n_layer = 6
        self.n_head = 8
        self.d_model = 256
        self.d_ff = 2048
        self.dropout = 0.1
        self.d_embed = 256

        gpt_config = GPT2Config(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.d_model,
            n_inner=self.d_ff,
            resid_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            max_position_embeddings=4096,
        )
        self.transformer_decoder = nn.ModuleList(
            [GPT2Block(gpt_config, layer_idx=i) for i in range(self.n_layer)])

        # positional encoding used for encoder
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # token embedding
        self.token_embedding = TokenEmbedding(self.n_token, self.d_embed,
                                              self.d_model)

        # output layer
        self.output_layer = nn.Linear(self.d_model, self.n_token)

        self.apply(weights_init)

    def forward(self, tgt):

        tgt = self.token_embedding(tgt)
        tgt = self.pos_encoding(tgt)

        for i in range(self.n_layer):
            tgt = self.transformer_decoder[i].forward(tgt)[0]

        out = self.output_layer(tgt)
        return out
