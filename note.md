## 0101 record

CODE REVIEW: `run_logistic_mkp.py`

1. what is `t`?
2. what is the MIP class? `mkp`: define the knapsack problem. 
3. code stricture

for each MIP scale (m,n,t)
    get pre-generated MIP dataset
    define mvbsolver

    train mvbsolver on train dataset and save model (.pkl file)
    
    for each test (single instance?)
        define grbmodel
        set mvbsolver with grbmodel, and add MVB cut
    
        solve problem using grbsolver -> original info
        solve problem using mvbsolver -> mvb info

4. whenIsBestObjFound?
5. how to **evaluate** the results?

CODE REVIEW: `logisticUtils.py`

1. define logisticTrainer and logisticPredictor.
2. for trainer, using sklearn.fit for each *variable*.

CONCLUSION

1. compare the time of finding the best solution: `OTime` and `MVBTime`, in log_all_warm file.
2. i think it is more convenient to write within the current framework.
3. only need to define `gnnUtils` and (`gnnTrainer`, `gnnPredictor`)

## 0102 record
1. graph data missing
2. graph generated in `data_generation.py`
3. `presolve` needs to be called to get graph
4. successfully generated graph data

PIPELINE
- [x] START: given problem class indset and network config
- [x] READ: instance from data/instances/indset
- [x] READ: graph from data/graphs/indset
- [x] LOAD: model from trained_models/indset/train_1000_4
- [x] PREDICT: probability (n,2) and prediction (n,)
- [x] SOLVE: original MIP instance
- [x] SOLVE: original MIP with warm start
- [x] MVB: add cut using probabilities given by gnn
- [x] SOLVE: mvb_model

1. redefine `getMultiVarBranch()` to receive external args. shape of Xpred? (n,1) probs[:,1]
2. COPT not supported? solved

## 0103 record

1. how to warm start copt by predictions. round and fill variable values. √
2. callback? -> use gurobi

- [x] warm start copt model
- [x] get solving time for each model
- [ ] use the best config of gnn

## 0105 record
- [x] gurobi license and test

## 0106 record
- [x] gurobi callback: copy model and use global vars, and use objnow and objbst for comparison
- [x] build experiment env
- [x] generate target dataset