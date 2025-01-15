#!/bin/bash
# python pipeline.py --prob_name indset --psucceed 0.9 --heuristics 0.05 --robust 1 --upCut 1  --lowCut 1 --gap 0.01
python pipeline.py --prob_name setcover --psucceed 0.99999 --heuristics 0.05 --robust 1 --upCut 1  --lowCut 1
# python pipeline.py --prob_name cauctions --psucceed 0.999999999999999 --heuristics 0.05 --robust 1 --upCut 0  --lowCut 1
# python pipeline.py --prob_name fcmnf
# python pipeline.py --prob_name gisp

# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions --dt_types val
# python data_generation.py --prob_name fcmnf --dt_types val
# python data_generation.py --prob_name gisp --dt_types val