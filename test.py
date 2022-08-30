import torch
import numpy as np
from torch.optim import Adam

from params_setting import settings
from data_preprocess import load_data
from model import MGCL
from train_AFGRL import train_the_model


args = settings()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = MGCL([128, 64], args)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Load data
data_o, data_s, data_a, train_loader, val_loader, test_loader = load_data(args)

train_the_model(model, optimizer, data_o, data_a, train_loader, val_loader, test_loader, args)