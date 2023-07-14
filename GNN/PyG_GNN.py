from torch_geometric.datasets import Planetoid

# 获取数据集
dataset = Planetoid(root='./tmp/Cora', name='Cora')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)



class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads = 1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, classes, )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


def trainAndEval(model, optimizer, data):
    #model = model_
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    # 评估
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'{model}:Accuracy: {acc:.4f}')



# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_GCN = GCN_Net(dataset.num_node_features, 64, dataset.num_classes).to(device)
model_SAGE = GraphSAGE_Net(dataset.num_node_features, 64, dataset.num_classes).to(device)
model_GAT = GAT_Net(dataset.num_node_features, 64, dataset.num_classes, 8).to(device)

data = dataset[0].to(device)
optimizer_GCN = torch.optim.Adam(model_GCN.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_SAGE = torch.optim.Adam(model_SAGE.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_GAT = torch.optim.Adam(model_GAT.parameters(), lr=0.01, weight_decay=5e-4)

trainAndEval(model_GCN, optimizer_GCN, data)
trainAndEval(model_SAGE, optimizer_SAGE, data)
trainAndEval(model_GAT, optimizer_GAT, data)










# model_GCN.train()
# for epoch in range(200):
#     optimizer_GCN.zero_grad()
#     out = model_GCN(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer_GCN.step()
# #评估
# model_GCN.eval()
# pred = model_GCN(data).argmax(dim=1)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')