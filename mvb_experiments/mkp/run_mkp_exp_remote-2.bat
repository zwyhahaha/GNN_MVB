cd result

python run_logistic_mkp_ano.py --ntest 10 --maxtime 600 --fixthresh 1.1 --psucceed 0.8 --warm 0 --gap 0.0005 >> 1-2.txt
python run_logistic_mkp_ano.py --ntest 10 --maxtime 600 --fixthresh 1.1 --psucceed 0.8 --warm 1 --gap 0.0005 >> 2-2.txt

python run_logistic_mkp_ano.py --ntest 10 --maxtime 600 --fixthresh 0.99 --psucceed 0.8 --warm 0 --gap 0.0005 >> 3-2.txt
python run_logistic_mkp_ano.py --ntest 10 --maxtime 600 --fixthresh 0.99 --psucceed 0.8 --warm 1 --gap 0.0005 >> 4-2.txt

