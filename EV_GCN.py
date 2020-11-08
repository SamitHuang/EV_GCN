import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn 
from PAE import PAE

class EV_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(EV_GCN, self).__init__()
        K=3       
        hidden = [hgc for i in range(lg)] 
        self.dropout = dropout
        self.edge_dropout = edge_dropout 
        bias = False 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.lg = lg 
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias)) 
        cls_input_dim = sum(hidden) 

        self.cls = nn.Sequential(
                torch.nn.Linear(cls_input_dim, 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.edge_net = PAE(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False): 
        if self.edge_dropout>0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0],1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask] 
            
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight)) 
        h0 = h
         
        for i in range(1, self.lg): 
            h = F.dropout(h, self.dropout, self.training)
            h= self.relu(self.gconv[i](h, edge_index, edge_weight)) 
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.cls(jk) 

        return logit, edge_weight

