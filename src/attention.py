import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % head_num == 0
        # number of hidden size after linear transform of EACH head
        self.d_k = int(hidden_size / head_num)
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(3)])
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    # calculate how likely the query and key, get w_i for each key_i. The result is sum_{w_i value_i}
    # the attention is calculated by DOT product
    def attention(self, query, key, value, mask: torch.Tensor = None, dropout: nn.Dropout = None):
        '''
        :param query: [batch_size, head_num, qlen, d_k]
        :param key: [batch_size, head_num, klen, d_k]
        :param value: [batch_size, head_num, klen, d_k]
        :param mask: [batch_size, 1, qlen, klen] or [batch_size, 1, 1, klen]
        :return: attention score: [batch_size, head_num, qlen, d_k]
        '''
        d_k = query.size(-1)
        # [batch_size, head_num, qlen, klen]
        matmul_res = torch.matmul(query, torch.transpose(key, -1, -2))
        norm_matmul = matmul_res / math.sqrt(d_k)
        if mask != None:
            norm_matmul = norm_matmul.masked_fill(mask == 0, -1e9)
        # softmax over key
        attn_scores = F.softmax(norm_matmul, dim=-1)

        if dropout != None:
            attn_scores = dropout(attn_scores)
        # [batch_size, head_num, qlen, d_k]
        attn_values = torch.matmul(attn_scores, value)

        return attn_scores, attn_values

    def forward(self, query, key, value, mask=None):
        '''
        :param query:[batch_size, len, hidden_size]
        :param key: [batch_size, len, hidden_size]
        :param value: [batch_size, len, hidden_size]
        :param mask: [batch_size, len, len] or [batch_size, 1, len]
        :return:
        '''
        batch_size = query.size(0)
        if mask != None:
            # the same for all h heads
            # [bz, h, len, len]
            mask = mask.unsqueeze(1)
        # do linear projections
        projects = []
        for input, linear in zip([query, key, value], self.linears):
            project = linear(input)
            project = project.view(batch_size, -1, self.head_num, self.d_k)
            project = torch.transpose(project, 1, 2)
            projects.append(project)
        pquery, pkey, pvalue = projects

        attn_values, self.attn_scores = self.attention(pquery, pkey, pvalue, mask=mask, dropout=self.dropout)
        # concatenate heads
        # [batch_size, qlen, head_num, d_k]
        attn_values = torch.transpose(attn_values, 1, 2).continuous()
        attn_values = attn_values.view(batch_size, -1, self.head_num * self.d_k)
        # [batch_size, qlen, hidden_size]
        attn_values = self.out_linear(attn_values)

        return attn_values