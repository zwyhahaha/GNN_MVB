#!/bin/bash
python pipeline.py --prob_name indset --psucceed 0.9 --heuristics 1.0
python pipeline.py --prob_name setcover --psucceed 0.95 --heuristics 1.0
# python pipeline.py --prob_name cauctions
# python pipeline.py --prob_name fcmnf
# python pipeline.py --prob_name gisp

# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions
# python data_generation.py --prob_name fcmnf
# python data_generation.py --prob_name gisp