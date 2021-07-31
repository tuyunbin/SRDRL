import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super(SelfAttention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.model.change_detector.att_dim, cfg.model.change_detector.att_head))
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class ChangeDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim
        self.feat_dim = cfg.model.change_detector.feat_dim
        self.att_head = cfg.model.change_detector.att_head
        self.att_dim = cfg.model.change_detector.att_dim

        self.img = nn.Linear(self.feat_dim, self.att_dim)

        self.SSRE = SelfAttention(cfg)

        self.context1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.context2 = nn.Linear(self.att_dim, self.att_dim)

        self.gate1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.gate2 = nn.Linear(self.att_dim, self.att_dim)

        self.dropout = nn.Dropout(0.5)

        self.embed = nn.Sequential(
            nn.Conv2d(self.att_dim*3, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.att_dim, 6)

    def forward(self, input_1, input_2):
        batch_size, C, H, W = input_1.size()
        input_1 = input_1.view(batch_size, C, -1).permute(0, 2, 1) # (128, 196, 1026)
        input_2 = input_2.view(batch_size, C, -1).permute(0, 2, 1)
        input_bef = self.img(input_1) # (128,196, 512)
        input_aft = self.img(input_2)

        input_bef = self.SSRE(input_bef, input_bef, input_bef)
        input_aft = self.SSRE(input_aft, input_aft, input_aft)

        input_diff = input_aft - input_bef

        input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
        input_bef_context = self.dropout(input_bef_context)
        input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
        input_bef_gate = self.dropout(input_bef_gate)
        input_befs = input_bef_gate * input_bef_context

        input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
        input_aft_context = self.dropout(input_aft_context)
        input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
        input_aft_gate = self.dropout(input_aft_gate)
        input_afts = input_aft_gate * input_aft_context

        input_bef = input_bef.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        input_aft = input_aft.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)

        input_befs = input_befs.permute(0,2,1).view(batch_size, self.att_dim, H, W)
        input_afts = input_afts.permute(0,2,1).view(batch_size, self.att_dim, H, W)
        input_diff = input_diff.permute(0,2,1).view(batch_size, self.att_dim, H, W)

        input_before = torch.cat([input_bef, input_diff, input_befs], 1)
        input_after = torch.cat([input_aft, input_diff, input_afts], 1)

        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = torch.sigmoid(self.att(embed_before))
        att_weight_after = torch.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_bef)
        attended_1 = (input_bef * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_aft)
        attended_2 = (input_aft * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)

        return pred, att_weight_before, att_weight_after, attended_1, attended_2, input_attended


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
