#!/bin/bash
# python pipeline.py --prob_name indset --psucceed_low 0.9 --psucceed_up 0.9 --heuristics 0.05 --robust 1 --upCut 1  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name setcover --psucceed_low 0.99 --psucceed_up 0.99999 --heuristics 0.05 --robust 1 --upCut 1  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name cauctions --psucceed_low 0.9999999999 --psucceed_up 0.9999999999999999999999999999 --heuristics 0.05 --robust 1 --upCut 1  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name fcmnf --psucceed_low 0.99999999999 --psucceed_up 0.9 --heuristics 0.05 --robust 0 --upCut 1  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name gisp

# python pipeline.py --prob_name indset --psucceed_low 0.9 --normalize 0 --psucceed_up 0.9 --heuristics 0.05 --robust 0 --upCut 1  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name setcover --psucceed_low 0.99 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name cauctions --psucceed_low 0.9999999 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name fcmnf --psucceed_low 0.9999999 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name gisp --psucceed_low 0.99999 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01

# python pipeline.py --prob_name fcmnf --psucceed_low 0.9999999999 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01
# python pipeline.py --prob_name fcmnf --psucceed_low 0.99999999999999 --normalize 0 --heuristics 0.05 --robust 0 --upCut 0  --lowCut 1 --gap 0.01

# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions --dt_types val
# python data_generation.py --prob_name fcmnf --dt_types target
# python data_generation.py --prob_name gisp --dt_types target

# caffeinate -i nohup python pipeline.py --prob_name gisp --sample 1 --data_free 1 --robust 0 --psucceed_low 0.99 --ratio_low 0.6 --ratio_up 0.0 --gap 0.05 --heuristics 1.0 --robust 0 --upCut 0  --lowCut 1 --ratio_involve 1 > output1.log 2>&1 &
caffeinate -i nohup python pipeline.py --prob_name gisp --sample 0 --data_free 1 --robust 0 --psucceed_low 0.99 --ratio_low 0.4 --ratio_up 0.0 --gap 0.05 --heuristics 1.0 --robust 0 --upCut 0  --lowCut 1 --ratio_involve 1 > output2.log 2>&1 &