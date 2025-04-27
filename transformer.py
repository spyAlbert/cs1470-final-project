import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Optional

class FeedForwardBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, activation=nnf.relu, drop_prob=0.):
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, bias=True, drop_prob=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5
        self.query_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.key_value_proj = nn.Linear(key_dim, query_dim * 2, bias=bias)
        self.output_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, queries, keys=None, mask=None):
        keys = keys if keys is not None else queries
        batch_size, seq_len, feature_dim = queries.shape
        _, ref_len, _ = keys.shape
        query_heads = self.query_proj(queries).reshape(batch_size, seq_len, self.num_heads, feature_dim // self.num_heads)
        key_value_heads = self.key_value_proj(keys).reshape(batch_size, ref_len, 2, self.num_heads, feature_dim // self.num_heads)
        keys, values = key_value_heads[:, :, 0], key_value_heads[:, :, 1]
        attention_scores = torch.einsum('bnhd,bmhd->bnmh', query_heads, keys) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(3), float('-inf'))
        attention_weights = attention_scores.softmax(dim=2)
        output = torch.einsum('bnmh,bmhd->bnhd', attention_weights, values).reshape(batch_size, seq_len, feature_dim)
        output = self.output_proj(output)
        return output, attention_weights


class TransformerBlock(nn.Module):
    def forward(self, x, y=None, mask=None):
        attention_output, attention_weights = self.attention(self.layer_norm1(x), y, mask)
        x = x + attention_output
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, attention_weights

    def __init__(self, input_dim, ref_dim, num_heads, hidden_ratio=4., use_bias=False, drop_prob=0., activation=nnf.relu,
                 normalization_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.layer_norm1 = normalization_layer(input_dim)
        self.attention = AttentionHead(input_dim, ref_dim, num_heads, bias=use_bias, drop_prob=drop_prob)
        self.layer_norm2 = normalization_layer(input_dim)
        self.feed_forward = FeedForwardBlock(input_dim, int(input_dim * hidden_ratio), activation=activation, drop_prob=drop_prob)


class TransformerNetwork(nn.Module):
    def forward(self, x, y=None, mask=None):
        for i, block in enumerate(self.blocks):
            if self.is_encoder_decoder and i % 2 == 0:
                x, _ = block(x, y)
            elif self.is_encoder_decoder:
                x, _ = block(x, x, mask)
            else:
                x, _ = block(x, y, mask)
        return x

    def __init__(self, input_dim: int, num_heads: int, num_layers: int, ref_dim: Optional[int] = None,
                 hidden_ratio: float = 2., activation=nnf.relu, normalization_layer: nn.Module = nn.LayerNorm, 
                 is_encoder_decoder: bool = False):
        super().__init__()
        ref_dim = ref_dim if ref_dim is not None else input_dim
        self.is_encoder_decoder = is_encoder_decoder
        if is_encoder_decoder:
            num_layers *= 2
        blocks = []
        for i in range(num_layers):
            if i % 2 == 0 and is_encoder_decoder:
                blocks.append(TransformerBlock(input_dim, ref_dim, num_heads, hidden_ratio, activation=activation, normalization_layer=normalization_layer))
            elif is_encoder_decoder:
                blocks.append(TransformerBlock(input_dim, input_dim, num_heads, hidden_ratio, activation=activation, normalization_layer=normalization_layer))
            else:
                blocks.append(TransformerBlock(input_dim, ref_dim, num_heads, hidden_ratio, activation=activation, normalization_layer=normalization_layer))
        self.blocks = nn.ModuleList(blocks)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.input_proj(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_embedding.unsqueeze(0).expand(x.shape[0], *self.prefix_embedding.shape)
        prefix = torch.cat((x, prefix), dim=1)
        output = self.transformer(prefix)[:, self.clip_length:]
        return output

    def __init__(self, clip_dim: int, embed_dim: int, prefix_len: int, clip_len: int, num_layers: int = 8):
        super().__init__()
        self.clip_length = clip_len
        self.transformer = TransformerNetwork(embed_dim, 8, num_layers)
        self.input_proj = nn.Linear(clip_dim, clip_len * embed_dim)
        self.prefix_embedding = nn.Parameter(torch.randn(prefix_len, embed_dim), requires_grad=True)
