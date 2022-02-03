import numpy as np
import torch
import os
import random
import torch.nn as nn

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)