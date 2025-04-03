from nn_utils import *

# Update constraint embeddings based on variable embeddings.
class GINVarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assigment, aggr='mean', dropout=0.0, activation='relu', norm='graph'):

        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
           
        super(GINVarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")

        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm 

        # Maps variable embeddings to scalar variable assigment.
        self.var_assigment = var_assigment

        # Learn joint representation of variable embedding and assignment.
        self.joint_var = Sequential(Linear(dim + 1, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        # Map edge features to embeddings with the same number of components as node embeddings.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.nn = Sequential(Linear(dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):
        batch_con_idx = batch_idx_tuple[1]
        var_assignment = self.var_assigment(source) 
        source = self.joint_var(torch.cat([source, var_assignment], dim=-1))
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)
        x = self.norm(x, batch_con_idx) if isinstance(self.norm, GraphNorm) else self.norm(x)
        x = self.dropout(x)
        x = self.nn((1 + self.eps) * target + x)

        return x

    def message(self, x_j, edge_attr):
        return self.act()(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


# Update variable embeddings based on constraint embeddings.
class GINConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
    
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))

        super(GINConVarBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")
        
        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm 

        # Map edge features to embeddings with the same number of components as node embeddings.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        # Learn joint representation of constraint embedding and error.
        self.joint_con = Sequential(Linear(dim*2, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.nn = Sequential(Linear(dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))


    def forward(self, source, target, error_con, edge_index, edge_attr, batch_idx_tuple, size):
        batch_var_idx = batch_idx_tuple[0]
        source = self.joint_con(torch.cat([source, error_con], dim=-1))
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)
        x = self.norm(x, batch_var_idx) if isinstance(self.norm, GraphNorm) else self.norm(x)
        x = self.dropout(x)
        x = self.nn((1 + self.eps) * target + x)

        return x

    def message(self, x_j, edge_attr):
        return self.act()(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINSimpleBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr='mean', dropout=0.0, activation='relu', norm='graph'):
    
        if aggr == 'comb':
            aggr = MultiAggregation(aggrs=['min', 'max', 'mean', 'std'], mode='proj', mode_kwargs=dict(in_channels=dim, out_channels=dim))
     
        super(GINSimpleBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")   
        
        self.act = activations[activation]
        Norm = NoNorm if isinstance(self.act(), SELU) else BatchNorm 

        # Map edge features to embeddings with the same number of components as node embeddings.
        self.edge_encoder = Sequential(PreNormLayer(edge_dim), Linear(edge_dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

        self.norm = normalizations[norm](dim)
        self.dropout = Dropout(dropout)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.nn = Sequential(Linear(dim, dim), self.act(), Linear(dim, dim), self.act(), Norm(dim))

    def forward(self, source, target, edge_index, edge_attr, batch_idx_tuple, size):
        var_batch_idx, con_batch_idx = batch_idx_tuple
        if size[1] == var_batch_idx.shape[0]:
            batch_idx = var_batch_idx
        elif size[1] == con_batch_idx.shape[0]:
            batch_idx = con_batch_idx
        else:
            raise Exception(f"{size},{var_batch_idx.shape[0]},{con_batch_idx.shape[0]}")
        edge_embedding = self.edge_encoder(edge_attr)
        x = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)
        x = self.norm(x, batch_idx) if isinstance(self.norm, GraphNorm) else self.norm(x)
        x = self.dropout(x)
        x = self.nn((1 + self.eps) * target + x)

        return x

    def message(self, x_j, edge_attr):
        return self.act()(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

GIN_bipartite_layers = {'varcon': GINVarConBipartiteLayer, 'convar': GINConVarBipartiteLayer, 'simple':GINSimpleBipartiteLayer}