from nn_utils import *

# Update constraint embeddings based on variable embeddings.
class SAGEVarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assigment, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
          
        super(SAGEVarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Maps variable embeddings to scalar variable assigment.
        self.var_assigment = var_assigment

        # Learn joint representation of variable embedding and assignemnt.
        self.joint_var = Sequential(Linear(dim + 1, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

        # Map edge features to embeddings with the same number of components as node embeddings.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

        self.lin_l = Linear(dim, dim, bias=True)
        self.lin_r = Linear(dim, dim, bias=False)
        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):
        con_batch_idx = batch_idx_tuple[1]
        var_assignment = self.var_assigment(source)
        source = self.joint_var(torch.cat([source, var_assignment], dim=-1))
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=source, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)
        out += self.lin_r(target)
        out = self.norm(out, con_batch_idx)
        out = self.dropout(out)
        #out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr):
        return torch.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# Update variable embeddings based on constraint embeddings.
class SAGEConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
          
        super(SAGEConVarBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Learn joint representation of constraint embedding and error.
        self.joint_con = Sequential(Linear(dim + dim, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

        self.lin_l = Linear(dim, dim, bias=True)
        self.lin_r = Linear(dim, dim, bias=False)
        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)

    def forward(self, source, target, error_con, edge_index, edge_attr, batch_idx_tuple, size):
        var_batch_idx = batch_idx_tuple[0]
        source = self.joint_con(torch.cat([source, error_con], dim=-1))
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=source, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)
        out += self.lin_r(target)
        out = self.norm(out, var_batch_idx)
        out = self.dropout(out)
        #out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr):
        return torch.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class SAGESimpleBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
         
        super(SAGESimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

        self.lin_l = Linear(dim, dim, bias=True)
        self.lin_r = Linear(dim, dim, bias=False)
        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):

        edge_embedding = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=source, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)
        out += self.lin_r(target)
        out = self.norm(out)
        out = self.dropout(out)
        #out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr):
        return torch.relu(x_j + edge_attr)

    def message_and_aggregate(self, adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return torch.matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


SAGE_bipartite_layers = {'varcon': SAGEVarConBipartiteLayer, 'convar': SAGEConVarBipartiteLayer, 'simple':SAGESimpleBipartiteLayer}