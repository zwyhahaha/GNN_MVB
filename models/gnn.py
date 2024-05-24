from nn_utils import *
from .edgeconv_bipartite_class import EC_bipartite_layers
from .gin_bipartite_class import GIN_bipartite_layers
from .sage_bipartite_class import SAGE_bipartite_layers
from graph_preprocessing import get_node_degrees

GNN_layers = {'EC': EC_bipartite_layers, 'GIN': GIN_bipartite_layers, 'SAGE':SAGE_bipartite_layers}

class BaseModel(torch.nn.Module):
    """
    Base model class, which implements pre-training methods.
    source: https://github.com/ds4dm/learn2branch-ecole/blob/main/model/model.py
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True
    
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden, activation='relu', norm='graph', dropout=0):
        super(MLP, self).__init__()
        self.act = activations[activation]
        self.norm = normalizations[norm]
        self.nn = Sequential(Linear(input_size, hidden),  self.norm(hidden), self.act(), Dropout(dropout), \
                             Linear(hidden, hidden),  self.norm(hidden), self.act(), Dropout(dropout), \
                             Linear(hidden, hidden),  self.norm(hidden), self.act(), Dropout(dropout) )
    
    def forward(self, x, batch_idx):
        
        for block in self.nn:
            
            if  isinstance(block, GraphNorm):
                x = block(x, batch_idx)
            else:
                x = block(x)

        return x

# Compute error signal.
class ErrorLayer(MessagePassing):
    def __init__(self, dim, var_assignment, *args, **kwargs):
        super(ErrorLayer, self).__init__(aggr="add", flow="source_to_target")
        self.var_assignment = var_assignment
        self.error_encoder = Sequential(Linear(1, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm(dim))

    def forward(self, h_var, edge_index, edge_attr, rhs, lb, ub, con_index, con_kind, con_degree, batch_idx_tuple, size):
        # Compute scalar variable assignment.
        assignment = self.var_assignment(h_var)
        assignment = assignment * (ub-lb) + lb
        out = self.propagate(edge_index, x=assignment, edge_attr=edge_attr, size=size)
        out = out - rhs
        out = self.error_encoder(out)
        out = softmax(out, con_index)

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out

class ViolationLayer(MessagePassing):
    def __init__(self, dim, var_assignment, dropout, activation, norm):
        super(ViolationLayer, self).__init__(aggr="add", flow="source_to_target")
        self.act1 = activations[activation]
        self.var_assignment = var_assignment
        self.error_encoder = Sequential(Linear(2, dim), self.act1(), Dropout(dropout), Linear(dim, dim))
        self.norm = normalizations[norm](dim)
        self.act2 = Tanh()

    def forward(self, h_var, edge_index, coeff, rhs, lb, ub, con_index, con_kind, con_degree, batch_idx_tuple, size):
        #Compute scalar variable assignment.
        assignment = self.var_assignment(h_var)
        assignment = assignment * (ub-lb) + lb
        Ax = self.propagate(edge_index, x=assignment, edge_attr=coeff, size=size)
        violation = Ax-rhs
        node_degree = get_node_degrees(edge_index, size)
        tmp = torch.cat([violation/node_degree, con_kind], dim=-1) 
        out = self.error_encoder(tmp)
        out = self.norm(out, batch_idx_tuple[1])
        out = self.act2(out)
        return out
    
    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out


class SimpleMIPGNN(BaseModel):
    def __init__(self, gnn_type, num_layers, var_feature_size, con_feature_size, hidden, dropout, aggr, activation='relu', norm='graph', binary_pred=False, **kwargs):
        super(SimpleMIPGNN, self).__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.binary_pred = binary_pred
        self.act = activations[activation]

        # Embed initial node features.
        self.var_node_encoder = Sequential(PreNormLayer(var_feature_size), Linear(var_feature_size, hidden), self.act(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(PreNormLayer(con_feature_size), Linear(con_feature_size, hidden), self.act(), Linear(hidden, hidden))

        # Bipartite GNN architecture.
        self.layers_con = torch.nn.ModuleList()
        self.layers_var = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            self.layers_con.append(GNN_layers[self.gnn_type]['simple'](1, hidden, aggr, 0))
            self.layers_var.append(GNN_layers[self.gnn_type]['simple'](1, hidden, aggr, 0))

        # Task layer
        self.mlp = MLP((num_layers + 1) * hidden, hidden, activation, norm, dropout)
        self.out_layer = Sequential(Linear(hidden, 2, bias=False))

        if isinstance(self.act(), SELU):
            self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_(self)

    def forward(self, batch):

        num_var_nodes = batch.num_var_nodes
        num_con_nodes = batch.num_con_nodes

        var_batch_idx, con_batch_idx = get_var_and_con_batch_idx(batch, num_var_nodes, num_con_nodes, DEVICE)
        batch_idx_tuple = (var_batch_idx, con_batch_idx)
    
        var_node_features = batch.var_node_features
        con_node_features = batch.con_node_features
        edge_index_var = batch.edge_index_var
        edge_index_con = batch.edge_index_con
        edge_features_var = batch.edge_features_var
        edge_features_con = batch.edge_features_con
        
        X_var = self.var_node_encoder(var_node_features)
        X_con = self.con_node_encoder(con_node_features)

        X_var_lst = [X_var]

        num_var = torch.sum(num_var_nodes)
        num_con = torch.sum(num_con_nodes)

        for i in range(self.num_layers):
            X_con = self.layers_var[i](X_var_lst[-1], X_con, edge_index_var, edge_features_var, batch_idx_tuple, (num_var, num_con))
            X_var = self.layers_con[i](X_con, X_var_lst[-1], edge_index_con, edge_features_con, batch_idx_tuple, (num_con, num_var))
            X_var_lst.append(X_var)

        X_var = torch.cat(X_var_lst, dim=-1)
        X_var = self.mlp(X_var, var_batch_idx)
        out = self.out_layer(X_var)

        return out
        
    def __repr__(self):
        return self.__class__.__name__


class MIPGNN(BaseModel):
    def __init__(self, gnn_type, num_layers, var_feature_size, con_feature_size, hidden, dropout, aggr, activation='relu', norm='graph', binary_pred=False, **kwargs):
        super().__init__()
        self.gnn_type, self.error_type = gnn_type.split("+")
        self.num_layers = num_layers
        self.binary_pred = binary_pred
        self.act = activations[activation]
        self.error_layer_cls = ErrorLayer if self.error_type == 'E' else ViolationLayer

        # Embed initial node features.
        self.var_node_encoder = Sequential(PreNormLayer(var_feature_size), Linear(var_feature_size, hidden), self.act(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(PreNormLayer(con_feature_size), Linear(con_feature_size, hidden), self.act(), Linear(hidden, hidden))

        # Compute variable assignement.
        self.layers_ass = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers_ass.append(Sequential(Linear(hidden, hidden), self.act(), Linear(hidden, 1), Sigmoid()))

        # Bipartite GNN architecture with error propagation
        self.layers_con = torch.nn.ModuleList()
        self.layers_var = torch.nn.ModuleList()
        self.layers_err = torch.nn.ModuleList()

        for i in range(self.num_layers):
            self.layers_con.append(GNN_layers[self.gnn_type]['convar'](1, hidden, aggr, 0, activation, norm))
            self.layers_var.append(GNN_layers[self.gnn_type]['varcon'](1, hidden, self.layers_ass[i], aggr, 0, activation, norm))
            self.layers_err.append(self.error_layer_cls(hidden, self.layers_ass[i], 0, activation, norm))
        
        # Task layer
        self.mlp = MLP((num_layers + 1) * hidden, hidden, activation, norm, dropout)
  
        self.out_layer = Sequential(Linear(hidden, 2, bias=False))
        
        if isinstance(self.act(), SELU):
            self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_(self)

    def forward(self, batch):
      
        num_var_nodes = batch.num_var_nodes
        num_con_nodes = batch.num_con_nodes

        var_batch_idx, con_batch_idx = get_var_and_con_batch_idx(batch, num_var_nodes, num_con_nodes, DEVICE)
        batch_idx_tuple = (var_batch_idx, con_batch_idx)

        var_node_features = batch.var_node_features
        con_node_features = batch.con_node_features
        edge_index_var = batch.edge_index_var
        edge_index_con = batch.edge_index_con
        edge_features_var = batch.edge_features_var
        edge_features_con = batch.edge_features_con
        rhs = batch.rhs
        con_index = batch.index_con
        con_degree = batch.con_node_features[:,-1].view(-1,1)
        con_kind = batch.con_kind
        lb, ub = batch.lb.view(-1,1), batch.ub.view(-1,1)
        
        X_var = self.var_node_encoder(var_node_features)
        X_con = self.con_node_encoder(con_node_features)

        X_var_lst = [X_var]

        num_var = torch.sum(num_var_nodes)
        num_con = torch.sum(num_con_nodes)
 
        for i in range(self.num_layers):
            X_err = self.layers_err[i](X_var_lst[-1], edge_index_var, edge_features_var, rhs, lb, ub, con_index, con_kind, con_degree, batch_idx_tuple, (num_var, num_con))
            X_con = self.layers_var[i](X_var_lst[-1], X_con, edge_index_var, edge_features_var, batch_idx_tuple, (num_var, num_con))
            X_var = self.layers_con[i](X_con, X_var_lst[-1], X_err, edge_index_con, edge_features_con, batch_idx_tuple, (num_con, num_var))
            X_var_lst.append(X_var)
        
        X_var = torch.cat(X_var_lst, dim=-1)

        X_var = self.mlp(X_var, var_batch_idx)
        out = self.out_layer(X_var)

        return out

    def __repr__(self):
        return self.__class__.__name__
