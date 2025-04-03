from nn_utils import *

# Variables to constrains.
class ECVarConBipartiteLayer(MessagePassing):

    def __init__(self, edge_dim, dim, var_assigment, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
      
        super(ECVarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")
  
        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm 
        
        # Maps variable embeddings to scalar variable assigment.
        self.var_assigment = var_assigment

        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        # Combine node and edge features of adjacent nodes.
        self.nn = Sequential(Linear(3 * dim + 1, dim), self.act(), Linear(dim, dim), self.act())

        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):
        con_batch_idx = batch_idx_tuple[1]
        var_assignment = self.var_assigment(source)
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, t=target, v=var_assignment, edge_attr=edge_embedding, size=size)
        x = self.norm(x, con_batch_idx) if not isinstance(self.norm, BatchNorm) else self.norm(x)
        x = self.dropout(x)
        
        return x

    def message(self, x_j, t_i, v_j, edge_attr):
        return self.nn(torch.cat([t_i, x_j, v_j, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class ECConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
        
        super(ECConVarBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm
        
        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))
        
        # Combine node, error, and edge features of adjacent nodes.
        self.nn = Sequential(Linear(4 * dim, dim), self.act(), Linear(dim, dim), self.act())
        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)
        
    def forward(self, source, target, error_con, edge_index, edge_attr, batch_idx_tuple, size):
        var_batch_idx = batch_idx_tuple[0]
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, t=target, e=error_con, edge_attr=edge_embedding, size=size)
        x = self.norm(x, var_batch_idx) if not isinstance(self.norm, BatchNorm) else self.norm(x)
        x = self.dropout(x)
        return x

    def message(self, x_j, t_i, e_j, edge_attr):
        return self.nn(torch.cat([t_i, x_j, e_j, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class ECSimpleBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
        
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
      
        super(ECSimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm
        
        # Maps edge features to the same number of components as node features.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))
        
        # Combine node and edge features of adjacent nodes.
        self.nn = Sequential(Linear(3 * dim, dim), self.act(), Linear(dim, dim), self.act())
        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):
        var_batch_idx, con_batch_idx = batch_idx_tuple
        if size[1] == var_batch_idx.shape[0]:
            batch_idx = var_batch_idx
        elif size[1] == con_batch_idx.shape[0]:
            batch_idx = con_batch_idx
        else:
            raise Exception(f"{size},{var_batch_idx.shape[0]},{con_batch_idx.shape[0]}")
        
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, t=target, edge_attr=edge_embedding, size=size)
        x = self.norm(x, batch_idx) if not isinstance(self.norm, BatchNorm) else self.norm(x)
        x = self.dropout(x)

        return x

    def message(self, x_j, t_i, edge_attr):
        return self.nn(torch.cat([t_i, x_j, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

EC_bipartite_layers = {'varcon': ECVarConBipartiteLayer, 'convar': ECConVarBipartiteLayer, 'simple':ECSimpleBipartiteLayer}