import time
import os
import os.path as osp
import networkx as nx
import itertools
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.mask import index_to_mask, mask_to_index
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
import torch
import cplex

from global_vars import *

def get_bipartite_graph(cpx_instance) -> nx.Graph:
    
    domain_map = dict(zip(['C', 'I', 'B', 'S', 'N'], ['continuous', 'integer', 'binary', 'semi_continuous', 'semi-integer']))

    n_vars = cpx_instance.get_stats().num_variables
    n_cons = cpx_instance.get_stats().num_linear_constraints

    var_names = cpx_instance.variables.get_names() 
    var_domains = list(map(domain_map.get, cpx_instance.variables.get_types()))
    var_lb = cpx_instance.variables.get_lower_bounds()
    var_ub = cpx_instance.variables.get_upper_bounds()
    obj_multiplier = cpx_instance.objective.get_sense() # 1 if min else -1
    obj_coeff = obj_multiplier * np.array(cpx_instance.objective.get_linear())

    con_names = cpx_instance.linear_constraints.get_names()
    shared_names = set(var_names) & set(con_names)
    for shared_name in shared_names:
        i = con_names.index(shared_name)
        con_names[i] = "constr_" + shared_name

    rhs = list(cpx_instance.linear_constraints.get_rhs())
    senses = list(cpx_instance.linear_constraints.get_senses())
    is_range_con = np.array(senses) == 'R'
    con_multiplier = np.where(np.array(senses) == 'G', -1, 1)
    changed_senses = np.where(np.array(senses) == 'E', 'E', 'L') 

    assert len(set(changed_senses) - {'E', 'L'}) == 0

    # For handling range constraints
    range_constraints = []
    for c_idx, is_range in enumerate(is_range_con):
        if is_range:
            range_value = cpx_instance.linear_constraints.get_range_values(c_idx) # which is equal to u - l
            rhs_val = rhs[c_idx]
            upper_bound = max(rhs_val + range_value, rhs_val)
            range_constraints.append(dict(c_idx = c_idx, con_name = con_names[c_idx], lower_bound = min(rhs_val + range_value, rhs_val)))
            rhs[c_idx] = upper_bound
    
    if len(range_constraints) > 0:
        print("#range constraints:", len(range_constraints))
    
    con_dict = dict(zip(range(n_cons), cpx_instance.linear_constraints.get_rows()))
    edge_names = list(itertools.chain.from_iterable(map(lambda item: [(con_names[item[0]], var_names[i]) for i in item[1].ind], con_dict.items())))
    edge_weights = list(itertools.chain.from_iterable(map(lambda item: con_multiplier[item[0]] * np.array(item[1].val), con_dict.items())))
    rhs = np.array(rhs) * con_multiplier
    G = nx.Graph()
    G.add_nodes_from(var_names+con_names)
    G.add_edges_from(edge_names)
    assert len(edge_weights) == len(edge_names)
    nx.set_edge_attributes(G, dict(zip(edge_names, edge_weights)), name='coeff')
    nx.set_node_attributes(G, dict(zip(var_names + con_names, [0]*n_vars + [1]*n_cons)), name="bipartite")
    nx.set_node_attributes(G, dict(zip(con_names, rhs)), name='rhs')
    nx.set_node_attributes(G, dict(zip(con_names, changed_senses)), name='kind')
    nx.set_node_attributes(G, dict(zip(var_names, var_lb)), name='lb')
    nx.set_node_attributes(G, dict(zip(var_names, var_ub)), name='ub')
    nx.set_node_attributes(G, dict(zip(var_names, var_domains)), name='domain')
    nx.set_node_attributes(G, dict(zip(var_names, obj_coeff)), name='obj_coeff')
    
    assert G.number_of_nodes() == n_vars + n_cons, (G.number_of_nodes(), n_vars, n_cons, np.sum(list(nx.get_node_attributes(G, 'bipartite').values())))
    
    for range_constr in range_constraints:
        
        c_idx = range_constr['c_idx']
        con_name = range_constr['con_name']
        rhs_val = range_constr['lower_bound'] * -1 
        new_con_name = con_name + "_left"

        G.add_node(new_con_name, bipartite = 1, rhs = rhs_val, kind = "L")

        var_idx_list = cpx_instance.linear_constraints.get_rows(c_idx).ind
        coef_list = cpx_instance.linear_constraints.get_rows(c_idx).val

        for var_idx, coeff in zip(var_idx_list, coef_list):
            G.add_edge(new_con_name, var_names[var_idx], coeff=coeff*-1)

    return G, con_names

def get_labeled_graph(G, instance_name, cpx_instance, solution_path):
    sol_pool_path = solution_path.joinpath(instance_name + "_pool.npz")

    var_names = cpx_instance.variables.get_names()
    obj_multiplier = cpx_instance.objective.get_sense()

    solutions_obj = np.load(sol_pool_path)['solutions']
    obj_vector = obj_multiplier * solutions_obj[:,0]
    incumbent_ind = np.argmin(obj_vector)
    norm_obj_vector = StandardScaler().fit_transform(obj_vector.reshape(-1,1)).ravel()
    solutions_matrix = solutions_obj[:,1:]
    
    incumbent_sol_vector = solutions_matrix[incumbent_ind]
    mean_bias_vector = np.mean(solutions_matrix, axis=0).ravel()
    sol_weights = np.array(torch.softmax(torch.from_numpy(-norm_obj_vector), 0)).reshape(-1,1)
    weighted_bias_vector = np.sum(solutions_matrix * sol_weights, 0).ravel()
    
    assert len(var_names) == weighted_bias_vector.shape[0]

    nx.set_node_attributes(G, dict(zip(var_names, incumbent_sol_vector)), name='incumbent')
    nx.set_node_attributes(G, dict(zip(var_names, mean_bias_vector)), name='mean_bias')
    nx.set_node_attributes(G, dict(zip(var_names, weighted_bias_vector)), name='weighted_bias')

    return G

# Preprocess indices of bipartite graphs to make batching work.
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_var_nodes, self.num_con_nodes]).view(2, 1)
        elif key in ['edge_index_con']:
            return torch.tensor([self.num_con_nodes, self.num_var_nodes]).view(2, 1)
        elif key in ['index_con']:
            return self.num_con_nodes
        elif key in ['index_var']:
            return self.num_var_nodes
        else:
            return 0

def create_data_object(instance_name, cpx_instance, graph, is_labeled=True, save_dir=None, preprocess_start_time=None):

    bipartite_vals = np.array(list(nx.get_node_attributes(graph, 'bipartite').values()))

   # Number of constraints.
    num_con_nodes = bipartite_vals.sum()

    # Number of variables.
    num_var_nodes = len(bipartite_vals) - num_con_nodes

    # Maps networkx ids to new variable node ids.
    var_names = cpx_instance.variables.get_names()
    assert len(var_names) == num_var_nodes, f"{len(var_names)}, {num_var_nodes}, {len(bipartite_vals)}"
    varnode_idx = dict(zip(var_names, range(num_var_nodes)))

    # Get constraint name list from the model and add range constraint names if any.
    con_names = list(cpx_instance.linear_constraints.get_names())
    con_senses = np.array(list(cpx_instance.linear_constraints.get_senses()))
    is_range_con = con_senses == 'R'
    num_range_cons = np.sum(is_range_con)
    assert len(con_names) + num_range_cons == num_con_nodes

    range_con_names = list(np.array(con_names)[is_range_con])
    range_con_names = [name+"_left" for name in range_con_names]
    con_names = con_names + range_con_names
    assert len(con_names) == num_con_nodes
    
    #  Maps networkx ids to new constraint node ids.
    connode_idx = dict(zip(con_names, range(num_con_nodes)))

    if is_labeled:
        # Targets
        y_real = pd.Series(index=range(num_var_nodes), dtype=float)
        y_norm_real = pd.Series(index=range(num_var_nodes), dtype=float)
        y_incumbent = pd.Series(index=range(num_var_nodes), dtype=float)
    
    relaxed_val = pd.Series(index=range(num_var_nodes), dtype=float)
    # Features for variable nodes.
    feat_var = pd.Series(index=range(num_var_nodes), dtype=object)

    obj = pd.Series(index=range(num_var_nodes), dtype=object)
    is_binary = pd.Series(index=range(num_var_nodes), dtype=bool)
    lb = pd.Series(index=range(num_var_nodes), dtype=float)
    ub = pd.Series(index=range(num_var_nodes), dtype=float)

    # Feature for constraints nodes.
    feat_con = pd.Series(index=range(num_con_nodes), dtype=object)
    # Right-hand sides of equations.
    rhs = pd.Series(index=range(num_con_nodes), dtype=object)

    # Kinds of equations: 1: <=, 0: ==
    con_kind = pd.Series(index=range(num_con_nodes), dtype=object)

    # Dual values for equations.
    dual_val = pd.Series(index=range(num_con_nodes), dtype=object)

    index_con = []
    index_var = []

    # Iterate over nodes, and collect features.
    for i, node_data in graph.nodes(data=True):
        # Node is a variable node.
        if node_data['bipartite'] == 0:
                
            index_var.append(0)
            
            isb = node_data['domain'] == 'binary'
            isc = node_data['domain'] == 'continuous'
            isi = node_data['domain'] == 'integer'
            
            is_binary[varnode_idx[i]] = isb
            lb_, ub_ = node_data['lb'], node_data['ub']
            
            assert ub_ >= lb_

            lb[varnode_idx[i]] = lb_
            ub[varnode_idx[i]] = ub_
  
            if is_labeled:
                w_bias = node_data['weighted_bias']
                incumbent = node_data['incumbent']

                y_real[varnode_idx[i]] = w_bias
                
                if abs(ub_ - lb_) > 1e-6:
                    norm_bias = (w_bias-lb_)/(ub_-lb_)
                    norm_incumbent = (incumbent-lb_)/(ub_-lb_)
                    
                    if not (0 <= norm_bias <= 1):
                        w_bias = np.clip(w_bias, lb_, ub_)
                        norm_bias = (w_bias-lb_)/(ub_-lb_)

                    y_norm_real[varnode_idx[i]] = norm_bias

                    if not (0 <= norm_incumbent <= 1):

                        incumbent = np.clip(incumbent, lb_, ub_)
                        norm_incumbent = (incumbent-lb_)/(ub_-lb_)

                    y_incumbent[varnode_idx[i]] = norm_incumbent
                
                else:
                    if ub_ == 0 and lb_ == 0:
                        y_norm_real[varnode_idx[i]] = 0
                        y_incumbent[varnode_idx[i]] = 0
                    else:
                        y_norm_real[varnode_idx[i]] = 1
                        y_incumbent[varnode_idx[i]] = 1
                    
                assert (0 <= y_incumbent[varnode_idx[i]] <= 1) & (0 <= y_norm_real[varnode_idx[i]] <= 1)# & (0 <= relaxed_val[varnode_idx[i]] <= 1)

            feat_var[varnode_idx[i]] = [int(isb), int(isc), int(isi), node_data['obj_coeff'], graph.degree[i]]
            obj[varnode_idx[i]] = [node_data['obj_coeff']]
            
        # Node is constraint node.
        elif node_data['bipartite'] == 1:
            
            i = i.replace('constr_', '')
            
            index_con.append(0)
            
            rhs_val = node_data['rhs']
            kind = node_data['kind'] == 'L'

            rhs[connode_idx[i]] = [rhs_val]
            con_kind[connode_idx[i]] = [float(kind)]
            feat_con[connode_idx[i]] = [float(kind), rhs_val, graph.degree[i]]
        
        else:
            print("Error in graph format.")
            exit(-1)

    # Edge list for var->con graphs.
    edge_list_var = []
    # Edge list for con->var graphs.
    edge_list_con = []

    # Create features matrices for variable nodes.
    edge_features_var = []
    # Create features matrices for constraint nodes.
    edge_features_con = []

    # Remark: graph is directed, i.e., each edge exists for each direction.
    # Flow of messages: source -> target.
    for s, t, edge_data in graph.edges(data=True):
        
        if graph.nodes[s]['bipartite'] == 1:
            con_name = s
            var_name = t
        else:
            con_name = t
            var_name = s
        con_name = con_name.replace('constr_', '')
    
        # Source node is constraint. C->V.
        edge_list_con.append([connode_idx[con_name], varnode_idx[var_name]])
        edge_features_con.append([edge_data['coeff']])

        # Source node is variable. V->C.
        edge_list_var.append([varnode_idx[var_name], connode_idx[con_name]])
        edge_features_var.append([edge_data['coeff']])
    
    # Create data object.
    data = BipartiteData()
    data.instance_name = instance_name
    data.obj = torch.tensor(obj, dtype=torch.float)
    data.is_binary = torch.tensor(is_binary, dtype=torch.bool)
    data.lb = torch.tensor(lb, dtype=torch.float)
    data.ub = torch.tensor(ub, dtype=torch.float)
    if is_labeled:
        data.y_real = torch.tensor(y_real, dtype=torch.float)
        data.y_norm_real = torch.tensor(y_norm_real, dtype=torch.float)
        data.y_incumbent = torch.tensor(y_incumbent, dtype=torch.float)
    data.var_node_features = torch.tensor(feat_var, dtype=torch.float)
    data.con_node_features = torch.tensor(feat_con, dtype=torch.float)
    data.rhs = torch.tensor(rhs, dtype=torch.float)
    data.con_kind = torch.tensor(con_kind, dtype=torch.float)
    data.edge_features_con = torch.tensor(edge_features_con, dtype=torch.float)
    data.edge_features_var = torch.tensor(edge_features_var, dtype=torch.float)
    data.num_var_nodes = torch.tensor(num_var_nodes)
    data.num_con_nodes = torch.tensor(num_con_nodes)
    data.edge_index_var =  torch.tensor(edge_list_var, dtype=torch.long).t().contiguous()
    data.edge_index_con = torch.tensor(edge_list_con, dtype=torch.long).t().contiguous()
    data.index_con = torch.tensor(index_con, dtype=torch.long)
    data.index_var = torch.tensor(index_var, dtype=torch.long)

    if is_labeled:
        Ax, violation = constraint_valuation(data.y_incumbent, data.edge_index_var, data.edge_features_var, data.rhs, data.lb, data.ub, data.con_kind, (data.num_var_nodes, data.num_con_nodes))
        data.Ax = Ax
    
        if violation.max() > 1e-5:
            print(">>>", instance_name, str(violation.max()))
        
    if preprocess_start_time:
        data.process_time = time.time() - preprocess_start_time

    if save_dir:
        torch.save(data, osp.join(save_dir, f'{instance_name}_data.pt'))
    
    print(instance_name, "data created in", round(time.time()-preprocess_start_time, 2), "seconds.")
    
    return data

# Preprocessing to create Torch dataset
class GraphDataset(InMemoryDataset):
    def __init__(self, prob_name,  dt_type, dt_name, instance_dir, graph_dir, instance_names, transform=None, pre_transform=None, pre_filter=None):
        self.prob_name = prob_name
        self.dt_name = dt_name
        self.dt_type = dt_type
        self.instance_dir = instance_dir
        self.graph_dir = graph_dir
        self.instance_names = instance_names

        super(GraphDataset, self).__init__(str(graph_dir), transform, pre_transform, pre_filter)
   
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.instance_names

    @property
    def processed_file_names(self):
        return [self.dt_name]

    def download(self):
        pass

    def process(self):

        data_list = []
        for i, instance_name in enumerate(self.instance_names):

            data_file = f'{instance_name}_data.pt'
            graph_path = self.graph_dir.joinpath(instance_name + "_labeled_graph.pkl")
            instance = self.instance_dir.joinpath(instance_name + INSTANCE_FILE_TYPES[self.prob_name])
            
            if data_file in os.listdir(self.processed_dir):
                print(data_file, "available")
                data = torch.load(self.processed_dir+ "/"+ data_file)
                
            else:
                cpx_instance = cplex.Cplex(str(instance))
                graph = nx.read_gpickle(graph_path)
                is_labeled = self.dt_type in ['train', 'val']
                data = create_data_object(instance_name, cpx_instance, graph, is_labeled, self.processed_dir)

            data_list.append(data)

        random.shuffle(data_list)
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(GraphDataset, self).get(idx)
        return idx, data


def scale_node_degrees(data_obj):
  
    idx, data = data_obj

    if 'is_transformed' in data:
        return data_obj

    if data.num_con_nodes > 0:
        norm_con_degree = node_degree_scaling(data.edge_index_var, (data.num_var_nodes, data.num_con_nodes))
        data.con_node_features[:, -1] = norm_con_degree.view(-1)

        norm_var_degree = node_degree_scaling(data.edge_index_con, (data.num_con_nodes, data.num_var_nodes))
        data.var_node_features[:, -1] = norm_var_degree.view(-1)
    else:
        data.var_node_features[:, -1] = 0
    
    data.is_transformed = True

    return idx, data

def node_degree_normalization(data):

    if data.num_con_nodes > 0:
        norm_con_degree = node_degree_scaling(data.edge_index_var, (data.num_var_nodes, data.num_con_nodes))
        data.con_node_features[:,-1] = norm_con_degree

        norm_var_degree = node_degree_scaling(data.edge_index_con, (data.num_con_nodes, data.num_var_nodes))
        data.var_node_features[:, -1] = norm_var_degree
    else:
        data.var_node_features[:, -1] = 0

    return data

def Abc_normalization(data):
        
    # Normalization of constraint matrix 
    norm_rhs, max_coeff = normalize_rhs(data.edge_index_var, data.edge_features_var, data.rhs, (data.num_var_nodes, data.num_con_nodes))
    data.rhs = norm_rhs
    data.con_node_features[:, 1] = norm_rhs.view(-1)
    data.edge_features_var /= max_coeff[data.edge_index_var[1]]
    data.edge_features_con /= max_coeff[data.edge_index_con[0]]

    # Normalization of objective coefficients 
    data.obj /= data.obj.abs().max()
    data.var_node_features[:,-2] = data.obj.view(-1)
    
    return data

def AbcNorm(data_obj):
    
    if isinstance(data_obj, tuple):
        idx, data = data_obj
    else:
        data = data_obj

    if 'is_transformed' in data:
        return data
    
    data = data.clone()

    # Normalizing A, b, and c coefficients
    data = Abc_normalization(data)

    # Node degree normalization
    data = node_degree_normalization(data)

    if 'dual_val' in data:
        data.dual_val /= data.dual_val.abs().max()

    if 'relaxed_sol_val' in data:
        data.relaxed_sol_val /= data.relaxed_sol_val.abs().max()
    
    data.is_transformed = True

    if isinstance(data_obj, tuple):
        return idx, data
    
    return data


class NormalizeRHS(MessagePassing):
    def __init__(self):
         super(NormalizeRHS, self).__init__(aggr="max", flow="source_to_target")

    def forward(self, edge_index, coeff, rhs, size):
        
        abs_coeff = self.propagate(edge_index, edge_attr=coeff, size=size)
        abs_rhs = torch.abs(rhs)
        max_coeff = torch.cat((abs_coeff, abs_rhs), dim=-1).max(dim=-1).values.view(-1,1)
        norm_rhs = rhs/max_coeff
        return norm_rhs, max_coeff

    def message(self, edge_attr):
        return torch.abs(edge_attr)

class NodeDegreeScaling(MessagePassing):
    def __init__(self):
         super(NodeDegreeScaling, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, edge_index, size):
        
        connected = torch.ones((size[0], 1), dtype=torch.float)
        node_degree = self.propagate(edge_index, connected=connected, size=size)
        norm_node_degree = node_degree / node_degree.max()
        
        return norm_node_degree.view(-1)
        
    def message(self, connected_j):
        return connected_j

class NodeDegreeCalculation(MessagePassing):
    def __init__(self):
         super(NodeDegreeCalculation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, edge_index, size):
        
        connected = torch.ones((size[0], 1), dtype=torch.float, device=DEVICE)
        total_degree = self.propagate(edge_index, connected=connected, size=size)

        return total_degree
        
    def message(self, connected_j):
        return connected_j

class ConstraintValuation(MessagePassing):
    def __init__(self):
        super(ConstraintValuation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, assignment, edge_index, coeff, rhs, lb, ub, con_kind, size):
        # con_kind = 1 for less than constraints (<=) and con_kind = 0 for equality constraints (=)
        if lb is None or ub is None:
            x = assignment
        else: # assignment is decision values normalized between lb and ub
            x = (assignment * (ub-lb) + lb).view(-1,1)
        Ax = self.propagate(edge_index, x=x, edge_attr=coeff, size=size)
        difference = Ax-rhs
        violation = torch.relu(difference) * con_kind + torch.abs(difference) * (1 - con_kind)

        return Ax, violation
    
    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out

class SumViolation(MessagePassing):
    def __init__(self):
        super(SumViolation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, violation, edge_index, size):

        output = self.propagate(edge_index, x=violation, size=size)

        return output
    
    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

normalize_rhs = NormalizeRHS()
node_degree_scaling = NodeDegreeScaling()
get_node_degrees = NodeDegreeCalculation()
constraint_valuation = ConstraintValuation()
sum_violation = SumViolation()