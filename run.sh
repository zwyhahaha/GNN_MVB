#!/bin/bash
python pipeline.py --prob_name indset
python pipeline.py --prob_name cauctions
python pipeline.py --prob_name fcmnf
python pipeline.py --prob_name gisp
python pipeline.py --prob_name setcover

# python data_generation.py --prob_name indset
# python data_generation.py --prob_name setcover
# python data_generation.py --prob_name cauctions
# python data_generation.py --prob_name fcmnf
# python data_generation.py --prob_name gisp