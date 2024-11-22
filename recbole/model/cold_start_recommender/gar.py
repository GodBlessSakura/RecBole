# -*- coding: utf-8 -*-
# @Time   : 2024/11/22 17:12
# @Author : yzh
# @Email  : 935878328@qq.com
# @File   : gar.py

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ColdStartRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss


class MetaEmbedding(ColdStartRecommender):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.

    """

    def __init__(self, config, dataset):
        super(MetaEmbedding, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.cross_layer_num = config["cross_layer_num"]
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.num_feature_field * self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(
                torch.zeros(self.num_feature_field * self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = (
            self.embedding_size * self.num_feature_field + self.mlp_hidden_size[-1]
        )

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.predict_layer = nn.Linear(in_feature_num, 1)
        self.reg_loss = RegLoss()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def forward(self, interaction):
        return

    def calculate_loss(self, interaction):
        return

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))