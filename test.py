import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from IPython import embed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse
import model
from tqdm import tqdm
import numpy as np
from utils import progress_bar
from utils import *
from models import *
from utils import model_train as train
from utils import model_test as test
import cvxpy as cp
from torch.utils.data import DataLoader
from calc_influence_function import calc_s_test_single_icml, calc_s_test_sgd
import scipy.io as scio
import joblib
import copy
import gc
dict_ = scio.loadmat('influence_score.mat') # 输出的为dict字典类型
print(dict_['data'].shape) # numpy.ndarray