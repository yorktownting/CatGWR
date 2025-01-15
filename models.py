import math

import torch
import torch.nn as nn
from math import ceil
import numpy as np


class CatGWRegression(torch.nn.Module):
    def __init__(self, num_of_nodes, num_of_heads, num_of_features, num_of_variables=0, spatial_weights=None,
                 vertex_y=None, vertex_x=None, ols_weights=None, add_skip_connection=True, bias=True,
                 dropout_prob=0.6):
        """
        :param num_of_nodes:
        :param num_of_heads:
        :param num_of_features:
        :param num_of_variables:
        :param spatial_weights:
        :param vertex_y:
        :param vertex_x:
        :param ols_weights:
        :param add_skip_connection:
        :param bias:
        :param dropout_prob:
        """

        super().__init__()

        layers = []

        gat_layer_1 = GATLayer(
            num_in_features=num_of_features,
            num_out_features=int(num_of_features / 2),
            num_of_heads=num_of_heads,
            spatial_weights=spatial_weights,
            add_skip_connection=True,
            dropout_prob=dropout_prob
        )
        layers.append(gat_layer_1)

        gatw_layer = GATWLayer(
            num_of_nodes=num_of_nodes,
            num_in_features=int(num_of_features / 2) * num_of_heads,
            num_of_heads=1,
            spatial_weights=spatial_weights,
            dropout_prob=dropout_prob
        )
        layers.append(gatw_layer)

        regression_layer = NNRegressionLayer(num_of_vertex=num_of_nodes,
                                             num_of_variables=num_of_variables,
                                             num_of_heads=num_of_heads,
                                             vertex_y=vertex_y,
                                             vertex_x=vertex_x,
                                             ols_weights=ols_weights,
                                             dropout_prob=dropout_prob
                                             )

        layers.append(regression_layer)

        self.gatw_net = nn.Sequential(
            *layers,
        )

    def forward(self, graph):
        return self.gatw_net(graph)


class GATLayer(torch.nn.Module):
    """
    Amplifer Module in CatGNWR, which is acually a GAT layer with spatial connectivity.
    The original GAT layer from the paper: Graph Attention Networks (https://arxiv.org/abs/1710.10903)
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, spatial_weights, concat=True,
                 activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.spatial_neighbors = torch.where(spatial_weights > 0, 0, -np.inf)

        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        self.batchnorm = nn.BatchNorm1d(num_in_features)

        self.multi_head_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar  , on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.sigmoid = nn.Sigmoid()
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.multi_head_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        # if not self.concat:
        #     # sum up the features' value, let shape = (N, FOUT) -> (N, 1)
        #     out_nodes_features = out_nodes_features.sum(dim=self.head_dim).unsqueeze(-1)

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #

        in_nodes_features, edge_distances = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert edge_distances.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={edge_distances.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)

        nodes_features_proj = self.multi_head_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)

        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + self.spatial_neighbors)

        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)

        return out_nodes_features, edge_distances


class GATWLayer(torch.nn.Module):
    """
    Contextualized spatial weighting Module
    """

    def __init__(self, num_of_nodes, num_in_features, num_of_heads, spatial_weights, dropout_prob=0.6):
        super().__init__()

        self.num_of_heads = num_of_heads
        num_feature_per_head = ceil(num_in_features / num_of_heads)
        self.num_feature_per_head = num_feature_per_head
        # functional layers
        self.multi_head_proj = nn.Linear(num_in_features, num_of_heads * num_feature_per_head, bias=False)
        # functional parameter
        # using x_t·target + x_s·source instead of [x_t|x_s]
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_feature_per_head))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_feature_per_head))

        # activation function
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.relu = nn.ReLU()

        # others
        self.dropout = nn.Dropout(dropout_prob)

        self.spatial_weights = spatial_weights
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.multi_head_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self, graph):
        #
        # Step 1: Project vertex features onto multiple heads and regularize them
        #
        in_node_features, edge_distances = graph
        num_of_nodes = in_node_features.shape[0]
        assert edge_distances.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={edge_distances.shape}.'

        in_node_features = self.dropout(in_node_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT)
        node_features_proj = self.multi_head_proj(in_node_features).view(-1, self.num_of_heads,
                                                                         self.num_feature_per_head)

        #
        # Step 2: Get similarity
        #
        scores_source = torch.sum((node_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((node_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)
        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast
        node_attention_scores = self.relu(scores_source + scores_target)

        # Step 3: integrate spatial weights
        multi_spatial_weights = node_attention_scores * self.spatial_weights

        return multi_spatial_weights


class NNRegressionLayer(torch.nn.Module):
    """
    Neural Network Regression Module.
    Get the regression coefficients for each node with the contextualized spatial weighting matirx, and make prediction.
    """
    def __init__(self, num_of_vertex, num_of_variables, num_of_heads,
                 vertex_y=None, vertex_x=None, ols_weights=None, dropout_prob=0.6):
        super().__init__()
        self.num_of_vertex = num_of_vertex
        self.num_of_variables = num_of_variables
        self.num_of_heads = num_of_heads

        # 功能层
        self.linear_beta_1 = nn.Linear(num_of_vertex, int(math.sqrt(num_of_vertex)) + 2 * num_of_variables, bias=True)
        self.linear_beta_2 = nn.Linear(int(math.sqrt(num_of_vertex)) + 2 * num_of_variables, 2 * num_of_variables,
                                       bias=True)
        self.linear_beta_3 = nn.Linear(2 * num_of_variables, num_of_variables, bias=True)

        self.vertex_y = vertex_y
        self.vertex_x = vertex_x
        self.ols_weights = ols_weights.repeat(num_of_vertex, 1)

        # 激活函数
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.ELU = nn.ELU()

        self.dropout = nn.Dropout(dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_beta_1.weight)
        nn.init.xavier_uniform_(self.linear_beta_2.weight)
        nn.init.xavier_uniform_(self.linear_beta_3.weight)

    def forward(self, data):
        multi_spatial_weights = data
        weights = self.dropout(multi_spatial_weights)
        weights = weights.permute(1, 2, 0).squeeze(-1)

        beta = self.linear_beta_1(weights)
        beta = self.linear_beta_2(beta)
        beta = self.linear_beta_3(beta)
        beta = self.leakyReLU(beta)
        beta = beta * self.ols_weights

        y_hat = torch.sum(beta * self.vertex_x, dim=-1)

        return (beta, y_hat, multi_spatial_weights)
