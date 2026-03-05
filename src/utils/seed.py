#Kiểm soát tính ngẫu nhiên của mô hình

import random
import numpy as np
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)