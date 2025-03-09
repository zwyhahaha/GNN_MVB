#!/bin/bash
# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions --dt_types val
# python data_generation.py --prob_name fcmnf --dt_types target
# python data_generation.py --prob_name gisp --dt_types target

# gnn
# caffeinate -i nohup python pipeline.py --prob_name indset --sample 0 --data_free 0 --robust 0 --tmvb 0.999 --psucceed_low 0.99 --psucceed_up 0.99 --gap 0.01 --heuristics 1.0 --robust 0 --upCut 1 --lowCut 1 --ratio_involve 0 > output.log 2>&1 &
# caffeinate -i nohup python pipeline.py --prob_name setcover --sample 0 --data_free 0 --robust 0 --tmvb 0.999 --psucceed_low 0.99 --psucceed_up 0.99 --gap 0.01 --heuristics 1.0 --robust 0 --upCut 1 --lowCut 1 --ratio_involve 0 > output.log 2>&1 &

# braching rule
# caffeinate -i nohup python pipeline.py --prob_name cauctions --sample 0 --data_free 0 --robust 1 --ratio_involve 1 --ratio_low 0.6 --ratio_up 0.0 --psucceed_low 0.999999999 --psucceed_up 0.0 --gap 0.001 --upCut 0 --lowCut 1 --heuristics 0.05 > output.log 2>&1 &

caffeinate -i nohup python pipeline.py --prob_name setcover --sample 0 --data_free 0 --robust 1 --fixratio 0.2 --ratio_involve 1 --ratio_low 0.4 --ratio_up 0.0 --psucceed_low 0.9 --gap 0.001 --upCut 0 --lowCut 1 --heuristics 0.05 > output.log 2>&1 &
wait $!
