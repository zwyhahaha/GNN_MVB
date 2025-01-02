from ml_augmented_opt import *

prob_name = 'indset'
train_dt_name = TRAIN_DT_NAMES[prob_name]
val_dt_name = VAL_DT_NAMES[prob_name]
target_dt_names_lst = TARGET_DT_NAMES[prob_name]
instance_name = 'instance_1'
target_dt_name = 'train_1000_4'

data_params = dict(
    random_state = [0],
    train_size = [1000],
    val_size = [200],
    test_size = [100],
    transfer_size = [100],
    prob_name = [prob_name]
)

model_hparams = dict(
    network_name = ['EC+V', 'EC', 'EC+E'],
    hidden = [32],
    num_layers = [8],
    dropout = [0.1],
    aggr = ['comb'],
    activation=['relu'],
    norm = ['graph'],
    binary_pred = [False],
    prenorm = [True],
    abc_norm = [True]
)

train_hparams = dict(
    batch_size = [8],
    num_epochs = [24],
    lr = [1e-4],
    weight_decay = [0.0],
    bias_threshold = [0.5],
    pred_loss_type = ['edl_digamma', 'bce'],
    edl_lambda = [6.0, None],
    evidence_func = ['softplus'],
    scheduler_step_size = [None],
    gamma = [None],
    scheduler_type = ['cycle']
)

#factors = ['network_name', 'strategy', 'abc_norm', 'prenorm']

param_grid = ParameterGrid({**model_hparams, **train_hparams, **data_params})

model_configs = []

for config in param_grid:

    if config['pred_loss_type'] == 'bce' and not config['edl_lambda'] is None:
        continue 
    if not config['pred_loss_type'] == 'bce' and config['edl_lambda'] is None:
        continue 
    if config['num_layers'] == 8 and config['hidden'] == 64:
        continue
    if config['num_layers'] == 4 and config['hidden'] == 32:
        continue

    model_configs.append(config)
    
config = model_configs[0]

data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')  
data = torch.load(str(data_path.joinpath(instance_name + "_data.pt")))
instance_name = data.instance_name
print(">>> Reading:", instance_name)

model_dir = PROJECT_DIR.joinpath('trained_models', prob_name, train_dt_name)
_, model, _ = load_model(config, model_dir, data)

probs, prediction, uncertainty, evidence, incumbent, binary_idx = get_prediction(config, model, data)
print(">>> Prediction:", prediction)
print(">>> Probability:", probs)

""" 
TODO:
Pipeline under construction
"""
import coptpy as cp 
from coptpy import COPT

env = cp.Envr()
cp_model = env.createModel(instance_name)
instance_path = 'data/instances/indset/train_1000_4/instance_1.lp'
initcpmodel = cp_model.read(instance_path)
cp_model.solve()

from mvb_experiments.MVB import *
from mvb_experiments.mkp.result.mkpUtils import *

m=4221
n=1000
mvbsolver = MVB(m, n)
mvbsolver.registerModel(initcpmodel, solver="copt")
mvbsolver.registerVars(list(range(n)))
mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs) # TODO: redefine this function
mvb_model.solve()
