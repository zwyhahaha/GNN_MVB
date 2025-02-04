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

## 0108 record
DATASET
1. `indset`: independent set, decide whether to include a node, such that an edge is not included, and maximize the cardinality
2. `setcover`: set covering, decide whether to include a set, such that the union is the universe, and minimize the number of sets
3. `cauctions`: combinatorial auctions, decide whether to allocate a subset of product to a buyer, and maximize the value
4. `fcmnf`: Fixed-Charge Multi-Commodity Network Flow: decide the flow, minimize the total cost with fixed charge
5. `gisp`: generalized independent set problem: divide into removable and unremovable edges. max the revenue - remove cost

- [x] fix min problem
- [x] set param Heuristics as 1

## 0109 record
- [x] run cauctions dataset: unbalanced upper and lower bound, and a hidden threshold in code? sort
- [ ] implement logistic regression model, how to generate the train dataset for logreg? perturb fcmnf
- [x] rerun setcover by removing the upper cut
- [x] solve four subproblems

## 0115 record
- [x] fix a bug in braching rule. >= a vs. <= (a-1)
- [x] use different probability for up and low cut

## 0116 record
- [x] The 3 subproblems run too fast? is there any problems? -> yes, copy from a model with constraints
- [ ] slower on dataset cauctions
- [ ] experiments design

## 0118 record
- [x] subproblems are hard to certify infeasibility
- [x] check the lp file, the constraints are correct; set the ksi value
- [ ] need to tune pSuccess value, or change the MIPGap.
- [x] run experiments for fcmnf, gisp: the predictions for these datasets are all 0. prediction model???

(0.9,0.999999)
5.202924966812134 4.4546167850494385 0.1715221405029297 45.79219579696655 [26.434773921966553, 6.3266448974609375, 12.561950922012329]
(0.9,0.999)
5.267876863479614 4.277310848236084 0.1682591438293457 33.014636754989624 [10.134559869766235, 4.9163899421691895, 17.477831840515137]
(0.999,0.9)
5.178627967834473 4.37123703956604 0.18824291229248047 31.716822862625122 [8.816731929779053, 0.21336698532104492, 22.168463945388794]
0.999
5.140126943588257 4.323051929473877 0.17703008651733398 29.24660587310791 [3.8896260261535645, 4.704403877258301, 17.315256118774414]
0.999999
5.208042144775391 4.490746021270752 0.17797112464904785 42.648179054260254 [7.831287145614624, 4.540284872055054, 24.10770010948181]
0.7
5.365480184555054 4.345301866531372 0.18372201919555664 17.468525886535645 [0.1407790184020996, 0.1584148406982422, 16.984760999679565]
5.160451889038086 4.398669004440308 0.18472599983215332 25.23917818069458 [1.2537739276885986, 0.8586239814758301, 22.940697193145752]
0.5 (gap for subproblems is smaller than gap/2, 447 vs 448, gap=0.5%)
5.023036003112793 4.526340961456299 3600.0 23.22577404975891 [0.13572001457214355, 0.1790010929107666, 22.748383045196533] 
transfer2000, gap=0, (0.999,0.9), cannot solve the problem?
12.451988935470581 12.305760860443115 1.1636438369750977 342.4629111289978 [237.63429808616638, 0.5387420654296875, 90.6875410079956]

fcmnf transfer (50 instances), >>1200s (14%)
Get MVB bounds...
0 0.0 80497 1013.1250758393608

gisp time (gap 0.05): 126.99090218544006 141.74243187904358

PRIMAL HEURISTIC
results/indset/valid_1000_4/gurobi_heuristics_0.05_fixthresh_1.1_psucceed_0.9_gap_0.01_maxtime_3600.0.csv

## 0119 record
- [x] for primal heuristic test, the distribution is skewed, so that pSucceed must be very high. add probs normalization to fix that.
- [ ] check the prediction acc in code, maybe there is a misnatch between prediction results and input
- [ ] add a loop for determine the time dominance. if 1, use callback time. if 0, use solving time.

CUDA_VISIBLE_DEVIVES=1 python ml_augmented_opt.py
compute in function `get_uncertainty_params`
main function:

```python
probs, pred, uncertainty, evidence, target, binary_idx = get_prediction(config, model, data) 
correct_pred_uncertainty = np.median(uncertainty[target==pred]) if threshold_type == 'median' else np.mean(uncertainty[target==pred])
val_u_mean_lst.append(correct_pred_uncertainty)
is_confident = (uncertainty <= correct_pred_uncertainty).ravel()
val_confident_ratio.append(is_confident.mean())
confident_acc = (pred[is_confident] == target[is_confident]).mean()
val_confident_acc_lst.append(confident_acc)
pred_sum = pred.sum()
incumbent_sum = target.sum()
val_bound_err_lst.append(incumbent_sum - pred_sum)
```
output
val_u_mean: 0.13467924 val_confident_ratio_mean: 0.44688 val_confident_acc_mean: 0.9986185287365168

## 0125 record
- [x] acc is high, but not informative
- [x] try different prediction models, tune the thresholds: not effective

## 0128 record
>>> Prediction accuracy: 0.871
>>> Prediction accuracy (1): 0.0
>>> Prediction accuracy (0): 0.9216931216833684
21 11.417043664575539 906 75.06402983581636
- 927 variables are involved in the MVB within interval [0.9, 1.1)
6.850820064544678 6.239526987075806 7.808387994766235 0.0 0.0

0 0.0 734 15.949459287184911
- 734 variables are involved in the MVB within interval [0.999999, 1.1)
Get MVB bounds...
0 0.0 411 11.934896128858549

## 0204 record
1. use higher threshold, fix lesser vars
2. data-free: use solution of LP relxation as prediction