import os
from pathlib import Path
from multiprocessing import cpu_count
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = torch.tensor(1e-8).to(DEVICE)

N_THREADS = cpu_count()

PROJECT_DIR = Path(os.path.abspath(
            os.path.join(
                os.path.abspath(__file__), ".."
            )
        ))

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "trained_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DT_NAMES = {
    'setcover' : 'train_500r_1000c_0.05d',
    'cauctions' : 'train_200_1000',
    'indset': 'train_1000_4',
    'fcmnf' : 'train',
    'gisp': 'train'
    }

VAL_DT_NAMES = {
    'setcover' : 'valid_500r_1000c_0.05d',
    'cauctions' : 'valid_200_1000',
    'indset': 'valid_1000_4',
    'fcmnf' : 'valid',
    'gisp': 'valid'
    }

TARGET_DT_NAMES = {
    'setcover' : ['test_500r_1000c_0.05d', 'transfer_1000r_2000c_0.05d', 'transfer_2000r_4000c_0.05d', 'transfer_4000r_8000c_0.05d', 'transfer_8000r_16000c_0.05d'],
    'cauctions' : ['test_200_1000', 'transfer_400_2000', 'transfer_800_4000', 'transfer_1600_8000', 'transfer_3200_16000'],
    'indset': ['test_1000_4', 'transfer_2000_4', 'transfer_4000_4', 'transfer_8000_4', 'transfer_16000_4'],
    'fcmnf' : ['test', 'transfer'],
    'gisp' : ['test', 'transfer']
    }

PROB_NAMES = list(TRAIN_DT_NAMES.keys())

INSTANCE_FILE_TYPES = {'setcover': '.lp', 'cauctions':'.lp', 'indset':'.lp', 'fcmnf':'.mps', 'gisp':'.mps'}

MODEL_INDEX = {'setcover': 0, 'cauctions': 0, 'indset':0, 'fcmnf':1, 'gisp':1}