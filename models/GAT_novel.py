from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

class VanillaGCN(torch.nn.Module):
    def __init__(self, region_size, features_in, features_out):
        super(VanillaGCN, self).__init__()
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization = torch.nn.BatchNorm1d(region_size)

    def forward(self,  features, Adjancy_Matrix):
        output = F.leaky_relu(
            self.batch_normalization(
                self.W(torch.matmul(Adjancy_Matrix, features))))
        return output


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, region_size, in_features_num, out_features_num, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features_num = in_features_num
        self.out_features_num = out_features_num
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = torch.nn.Parameter(torch.zeros(size=(in_features_num, out_features_num)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_features_num, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.region_size = region_size
        self.fc_reduce = nn.Linear(region_size * region_size, region_size)
        self.fc_pad = nn.Linear(region_size, region_size * region_size)

        self.sparse = torch.nn.Parameter(torch.zeros(size=(region_size, region_size)))
        torch.nn.init.xavier_uniform_(self.sparse.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()
        self.batch_normalization_layer = torch.nn.BatchNorm1d(out_features_num)

    def min_max_normalization(self, tensor):
        B = tensor.size()[0]
        min_values_dim1, _ = torch.min(tensor, dim=1)
        min_val, _ = torch.min(min_values_dim1, dim=1)

        max_values_dim1, _ = torch.max(tensor, dim=1)
        max_val, _ = torch.max(max_values_dim1, dim=1)

        min_values_broadcasted = min_val.view(B, 1, 1)
        max_values_broadcasted = max_val.view(B, 1, 1)

        matrix_normalized = (tensor - min_values_broadcasted) / (max_values_broadcasted - min_values_broadcasted)
        matrix_normalized = matrix_normalized * 30 - 20
        return matrix_normalized

    def forward(self, inp, adj):
        batch_size, N = inp.size()[0], inp.size()[1]
        hidden_features = torch.matmul(inp, self.W)

        adj_input = torch.cat([hidden_features.repeat(1, 1, N).view(inp.size()[0], N*N, -1), hidden_features.repeat(1, N, 1)], dim=2).view(inp.size()[0], N, -1, 2*self.out_features_num)
        
        e = self.leakyrelu(torch.matmul(adj_input, self.a).squeeze(3))
        e = self.min_max_normalization(e)
        attention = self.sigmoid(e)

        attention = F.dropout(attention, self.dropout,
                              training=self.training)
    
        h_prime = torch.matmul(attention, hidden_features)
        if self.concat:
            h_prime = F.elu(h_prime)

        return h_prime, attention

class whole_network(torch.nn.Module):
    def __init__(self, region_size, in_features_num, hidden_num1, hidden_num2, dropout):
        super(whole_network, self).__init__()

        self.correlation = None
        self.in_features_num = in_features_num
        self.hidden_num1 = hidden_num1
        self.hidden_num2 = hidden_num2
        self.W1 = torch.nn.Parameter(torch.zeros(
            size=(self.in_features_num, self.hidden_num1)))
        self.W2 = torch.nn.Parameter(torch.zeros(
            size=(self.hidden_num1, self.hidden_num2)))  

        self.region_size = region_size
        self.dropout = dropout
        self.batch_normalization_input = torch.nn.BatchNorm1d(in_features_num)

        self.gat_layer_1 = GraphAttentionLayer(region_size=self.region_size,
            in_features_num=self.in_features_num, out_features_num=self.hidden_num1, dropout=self.dropout, alpha=0.01, concat=True)

        self.gat_layer_2 = GraphAttentionLayer(region_size=self.region_size,
            in_features_num=self.hidden_num1, out_features_num=self.hidden_num2, dropout=self.dropout, alpha=0.01, concat=False)

        self.gcn_layer_3 = VanillaGCN(region_size, self.hidden_num2, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x, edge_weight):
        gat_output_1, fc1 = self.gat_layer_1(x, edge_weight)
        gat_output_2, fc2 = self.gat_layer_2(gat_output_1, fc1)
        gcn_output_3 = self.gcn_layer_3(gat_output_2, fc2)

        return gcn_output_3, fc2

class GAT_novel(nn.Module):
    def __init__(self, region_size, in_feature_dim, hidden_dims1, hidden_dims2, dropout):
        super(GAT_novel, self).__init__()
        self.gcns = whole_network(region_size, in_feature_dim, hidden_dims1, hidden_dims2, dropout)
        self.fc = nn.Linear(region_size, 2)
        self.softmax = nn.Softmax(dim=1)        

    def forward(self, x_Pearson, adj_weights):
        x, attention = self.gcns(x_Pearson, adj_weights)
        flattened_x = x.view(x.shape[0], -1)
        x = self.fc(flattened_x)
        x = self.softmax(x)        
        return x, attention