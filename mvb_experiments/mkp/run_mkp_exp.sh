# #!/bin/bash
# #BSUB -J mkp-test
# #BSUB -e /nfsshare/home/gaowenzhi/Desktop/mkp.log
# #BSUB -o /nfsshare/home/gaowenzhi/Desktop/mkp.txt
# #BSUB -n 4
# #BSUB -q cauchy
# #BSUB -m cauchy01

# conda activate P37
cd result

rm *.txt
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.7 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.7 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.8 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.8 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.9 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.9 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.7 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.7 --warm 1 --gap 0.01

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.8 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.8 --warm 1 --gap 0.01

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.9 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 1.1 --psucceed 0.9 --warm 1 --gap 0.01

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.7 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.7 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.8 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.8 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.9 --warm 0 --gap 0.0005
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.9 --warm 1 --gap 0.0005

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.7 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.7 --warm 1 --gap 0.01

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.8 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.8 --warm 1 --gap 0.01

python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.9 --warm 0 --gap 0.01
python run_logistic_mkp.py --ntest 5 --maxtime 120 --fixthresh 0.95 --psucceed 0.9 --warm 1 --gap 0.01