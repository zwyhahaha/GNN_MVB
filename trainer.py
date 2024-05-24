import os
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import time

from torch_geometric.loader import DataLoader
from sklearn.model_selection import ParameterGrid
import wandb

from global_vars import *
from utils import *
from loss import scoring

EPS = torch.tensor(1e-8).to(DEVICE)
zero = torch.tensor(0).to(DEVICE)
one = torch.tensor(1).to(DEVICE)
GLOBAL_STEP = None

def pretrain(model, pretrain_loader):
    model.pre_train_init()
    i = 0
    while True:
        for batch_idx, (graph_idx, batch) in enumerate(tqdm(pretrain_loader)):
            batch.to(DEVICE)
            if not model.pre_train(batch):
                break

        if model.pre_train_next() is None:
            break
        i += 1
    return i

def step(epoch, model, loader, optimizer, scheduler, criterion, step_type, \
        bias_threshold=0.5, binary_pred=False,
        eval=True, print_log=True, **kwargs):

    global GLOBAL_STEP
    model.eval() if eval else model.train()

    if step_type == 'train' and not eval:
        print("Training...", GLOBAL_STEP)
    elif step_type == 'train' and eval:
        print("Evaluating training set...")
    elif step_type == 'val':
        print("Validating...")
    
    loss_all =  torch.tensor(0.0).to(DEVICE)
    bias_tuples = []
    evidence_tuples = []
    
    for batch_idx, (graph_idx, batch) in enumerate(tqdm(loader)):
        free_gpu_memory()

        batch = batch.to(DEVICE)       
        y = torch.where(batch.y_incumbent <= bias_threshold, zero, one).to(DEVICE)
  
        output = model(batch)

        loss, evidence_tuple, uncertainty = criterion(GLOBAL_STEP, graph_idx, batch, output, y, binary_pred, step_type)

        if not eval:
            loss.backward()
            optimizer.step()
            
            if type(scheduler) in [NoamLR, lr_scheduler.OneCycleLR]:
                scheduler.step()
            
            optimizer.zero_grad()
            
            GLOBAL_STEP += 1

        loss_all += loss.item()

        evidence_tuples.append(evidence_tuple)
            
        pred = torch.softmax(output, dim=-1)[:,1].view(-1)

        if binary_pred:
            pred = pred[batch.is_binary]   
            y = y[batch.is_binary]
            #uncertainty = uncertainty[batch.is_binary]

        pred_all = torch.cat([pred_all, pred]) if batch_idx > 0 else pred
        y_all = torch.cat([y_all, y]) if batch_idx > 0 else y
        uncertainty_all = torch.cat([uncertainty_all, uncertainty]) if batch_idx > 0 else uncertainty

        true_bias = torch.mean(y.to(torch.float))
        pred_bias = torch.mean(pred.round())
        soft_pred_bias = torch.mean(pred)
        bias_tuples.append([true_bias, pred_bias, soft_pred_bias, torch.abs(true_bias-pred_bias)])

    if not eval and type(scheduler) is lr_scheduler.StepLR:
        scheduler.step()
 
    lr = scheduler.optimizer.param_groups[0]['lr']
    log = scoring(batch_idx, loss_all, bias_tuples, evidence_tuples, uncertainty_all, pred_all, y_all, lr, bias_threshold, step_type, print_log)

    return log

def train(model_name, model, criterion, optimizer, scheduler, pretrain_loader, train_loader, val_loader, \
          config, WANDB_LOG=False, model_dir=''):
    
    global GLOBAL_STEP
    GLOBAL_STEP = 0

    print(">> Training starts on the current device", DEVICE)

    if WANDB_LOG:
        run = wandb.init(project='gnn4co', config=config, force=True, name=model_name)

    if config['prenorm']:
        print(">> Pretraining for prenorm...")
        pretrain(model, pretrain_loader)

    for epoch in range(1, config['num_epochs']+1):
        print(">> Epoch", epoch, ''.join(["-"]*100))
        epoch_time = time.time()
   
        train_log = step(epoch, model, train_loader, optimizer, scheduler, criterion, 'train', eval=False, **config)

        with torch.no_grad():
            free_gpu_memory()
            val_log = step(epoch, model, val_loader, optimizer, scheduler, criterion, 'val',  eval=True, **config)
            
            if epoch == 1 or round(val_log['val_acc'], 3) >= best_val_score:
                
                best_val_score = round(val_log['val_acc'], 3) 
                              
                if model_dir:
                    torch.save(model.state_dict(), str(model_dir.joinpath(model_name + ".pt")))

        epoch_time = time.time() - epoch_time
        log = {"epoch": epoch, "epoch_time": epoch_time, **train_log,  **val_log}
        
        if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_log['val_acc'])

        if WANDB_LOG:
            wandb.log(log)    
            
    if WANDB_LOG:
        run.finish()

    return model

def main(train_dt, val_dt, config, WANDB_LOG):
    
    if WANDB_LOG:
        wandb.login(force=True)

    model_dir = PROJECT_DIR.joinpath('trained_models', config["prob_name"], train_dt.dt_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
        
    seed = config['random_state']
    batch_size = config["batch_size"]
    
    set_random_state(seed)
    
    pretrain_loader = DataLoader(train_dt, batch_size=batch_size*2, shuffle=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dt, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
     
    var_feature_size = train_dt[0][1].var_node_features.size(-1)
    con_feature_size = train_dt[0][1].con_node_features.size(-1)
    model_name, model, criterion, optimizer, scheduler = get_model(model_dir, var_feature_size, con_feature_size, len(train_loader), **config)

    if Path(model_dir.joinpath(model_name + ".pt")).exists():
        print(f">> {model_name} available in {model_dir}")
        return
    
    model = train(model_name, model, criterion, optimizer, scheduler, pretrain_loader, train_loader, val_loader, config, WANDB_LOG, model_dir)

    return model

if __name__ == '__main__':

    datasets = {
    'setcover' : ['train_500r_1000c_0.05d', 'valid_500r_1000c_0.05d'],
    'cauctions' : ['train_200_1000', 'valid_200_1000'],
    'indset' : ['train_1000_4', 'valid_1000_4'],
    'fcmnf': ['train', 'valid'],
    'gisp': ['train', 'valid']}

    prob_names = ['setcover', 'cauctions', 'indset', 'fcmnf', 'gisp']
    prob_name = prob_names[0]
    dt_names = datasets[prob_name]
    
    WANDB_LOG = True

    data_params = dict(
        random_state = [0],
        train_size = [1000],
        val_size = [200],
        prob_name = [prob_name]
    )

    model_params = dict(
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

    train_params = dict(
        batch_size = [8],
        num_epochs = [24],
        lr = [1e-4],
        weight_decay = [0.0],
        bias_threshold = [0.5],
        pred_loss_type = ['edl_digamma', 'bce'],
        edl_lambda = [1.0, None],
        evidence_func = ['softplus'],
        scheduler_step_size = [None],
        gamma = [None],
        scheduler_type = ['cycle']
    )

    param_grid = ParameterGrid({**model_params, **train_params, **data_params})
    print(">> Parameter grid size:",len(param_grid))

    last_abc_norm_config = None
    for seed in range(1):
        for ID, config in enumerate(param_grid):
            config["random_state"] = seed
            print(">>>>",seed, ID, (ID+1)/len(param_grid))
            pprint(config)
            
            if ID == 0 or config['abc_norm'] != last_abc_norm_config:
                dt_sizes = [config['train_size'], config['val_size']]
                train_dt, val_dt = get_co_datasets(prob_name, dt_names, dt_sizes, config['abc_norm'])
                last_abc_norm_config = config['abc_norm']

            config['train_size'] = len(train_dt)
            config['val_size'] = len(val_dt)

            if config['pred_loss_type'] == 'bce' and not config['edl_lambda'] is None:
                continue 
            if not config['pred_loss_type'] == 'bce' and config['edl_lambda'] is None:
                continue 
                     
            main(train_dt, val_dt, config, WANDB_LOG=WANDB_LOG)