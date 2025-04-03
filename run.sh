#!/bin/bash

for heuristics in 0 0.05 1.0; do
    # gnn
    python gnn_experiments.py --prob_name indset --solver gurobi --dt_name transfer_2000_4 --data_free 0 --robust 0 --tmvb 0.9 --psucceed_low 0.99 --psucceed_up 0.99 --heuristics $heuristics
    python gnn_experiments.py --prob_name setcover --solver gurobi --data_free 0 --robust 0 --tmvb 0.9 --psucceed_low 0.999 --psucceed_up 0.99999 --heuristics $heuristics
    python gnn_experiments.py --prob_name cauctions --solver gurobi --data_free 0 --robust 0 --ratio_involve 1 --ratio_up 0.0 --ratio_low 0.05 --psucceed_low 0.999 --heuristics $heuristics

    # data free
    python gnn_experiments.py --prob_name cauctions --solver gurobi --data_free 1 --robust 0 --tmvb 0.8 --psucceed_low 0.9 --psucceed_up 0.999 --heuristics $heuristics
    python gnn_experiments.py --prob_name setcover --solver gurobi --data_free 1 --robust 0 --tmvb 0.9999 --psucceed_low 0.9999 --psucceed_up 0.999 --heuristics $heuristics

    # branching rule
    python gnn_experiments.py --prob_name cauctions --solver gurobi --sample 0 --data_free 0 --robust 1 --ratio_involve 1 --ratio_low 0.6 --ratio_up 0.0 --psucceed_low 0.999999999 --psucceed_up 0.0 --gap 0.001 --upCut 0 --lowCut 1 --heuristics $heuristics
    python gnn_experiments.py --prob_name setcover --solver gurobi --sample 0 --data_free 0 --robust 1 --fixratio 0.2 --ratio_involve 1 --ratio_low 0.4 --ratio_up 0.0 --psucceed_low 0.9 --gap 0.001 --upCut 0 --lowCut 1 --heuristics $heuristics
done

for heuristics in 0 0.05 1.0; do
    # gnn
    python gnn_experiments.py --prob_name indset --solver copt --dt_name transfer_2000_4 --data_free 0 --robust 0 --tmvb 0.9 --psucceed_low 0.99 --psucceed_up 0.99 --heuristics $heuristics
    python gnn_experiments.py --prob_name setcover --solver copt --data_free 0 --robust 0 --tmvb 0.9 --psucceed_low 0.999 --psucceed_up 0.99999 --heuristics $heuristics
    python gnn_experiments.py --prob_name cauctions --solver copt --data_free 0 --robust 0 --ratio_involve 1 --ratio_up 0.0 --ratio_low 0.05 --psucceed_low 0.999 --heuristics $heuristics

    # data free
    python gnn_experiments.py --prob_name cauctions --solver copt --data_free 1 --robust 0 --tmvb 0.8 --psucceed_low 0.9 --psucceed_up 0.999 --heuristics $heuristics
    python gnn_experiments.py --prob_name setcover --solver copt --data_free 1 --robust 0 --tmvb 0.9999 --psucceed_low 0.9999 --psucceed_up 0.999 --heuristics $heuristics
done