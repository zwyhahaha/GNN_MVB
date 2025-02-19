#!/bin/bash
# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions --dt_types val
# python data_generation.py --prob_name fcmnf --dt_types target
# python data_generation.py --prob_name gisp --dt_types target

# gnn
# caffeinate -i nohup python pipeline.py --prob_name indset --sample 0 --data_free 0 --robust 0 --tmvb 0.999 --psucceed_low 0.99 --psucceed_up 0.99 --gap 0.01 --heuristics 1.0 --robust 0 --upCut 1 --lowCut 1 --ratio_involve 0 > output.log 2>&1 &
# caffeinate -i nohup python pipeline.py --prob_name setcover --sample 0 --data_free 0 --robust 0 --tmvb 0.999 --psucceed_low 0.99 --psucceed_up 0.99 --gap 0.01 --heuristics 1.0 --robust 0 --upCut 1 --lowCut 1 --ratio_involve 0 > output.log 2>&1 &


# data free
# caffeinate -i nohup python pipeline.py --prob_name cauctions --sample 0 --data_free 1 --robust 0 --tmvb 0.8 --psucceed_low 0.9 --psucceed_up 0.999 --gap 0.001 --heuristics 1.0 --robust 0 --upCut 1  --lowCut 1 --ratio_involve 0 > output.log 2>&1 &
caffeinate -i nohup python pipeline.py --prob_name fcmnf --sample 0 --data_free 1 --robust 0 --tmvb 0.9999 --psucceed_low 0.9999 --psucceed_up 0.99999999999 --gap 0.001 --heuristics 1.0 --robust 0 --upCut 1  --lowCut 1 --ratio_involve 0 > output.log 2>&1 &
# caffeinate -i nohup python pipeline.py --prob_name setcover --sample 0 --data_free 1 --robust 0 --psucceed_low 0.9999 --psucceed_up 0.999 --tmvb 0.9999 --gap 0.01 --heuristics 1.0 --robust 0 --upCut 1 --lowCut 1 --ratio_involve 0 > output.log 2>&1 &