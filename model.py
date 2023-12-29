# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
          input_size = 256
        self.dense = nn.Linear(input_size, 256)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(256, 2)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class GNNReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        self.w_embeddings = self.encoder.bert.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer
        self.gnn = ReGGNN(feature_dim_size=768,
                            hidden_size=256,
                            num_GNN_layers=2,
                            dropout=config.hidden_dropout_prob,
                            residual=not False,
                            att_op="mul")
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None):
        # construct graph
        adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=3)
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        logits = self.classifier(outputs)
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob