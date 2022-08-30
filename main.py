import torch
import numpy as np

from params_setting import settings
from data_preprocess import load_data
from CSGNN import Create_model
from train_CSGNN import train_model


args = settings()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
data_o, data_s, data_a, train_loader, val_loader, test_loader = load_data(args)


# train and test the CSGNN model
model, optimizer = Create_model(args)
train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args)
