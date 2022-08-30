import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from model import MGCL

def train_the_model(model, optimizer, data_o, data_a, train_loader, val_loader, test_loader, args):
    loss_fct = nn.BCELoss()
    m = torch.nn.Sigmoid()
    loss_history = []
    max_auc = 0
    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_a.to('cuda')

    # Train the AFGRL model
    t_total = time.time()
    lbl = data_a.y
    model_max = copy.deepcopy(model)
    print('Start training...')
    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + '--------')
        y_pred_train, y_label_train = [], []
        for i, (label, inp) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            pred, ind, k, output, loss2 = model(data_o.x, data_a.y, data_o.edge_index, inp)
            log = torch.squeeze(m(output))
            loss1 = loss_fct(log, label.float())
            loss = loss1 + 0.2 * loss2

            loss_history.append(loss)
            loss.backward()
            optimizer.step()
            model.update_moving_average()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        if not args.fastmode:
            roc_val, prc_val, f1_val, loss_val = test(model, val_loader, data_o, data_a, args)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val

            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'auroc_val: {:.4f}'.format(roc_val),
                  'auprc_val: {:.4f}'.format(prc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    loss_history = [x.cpu().detach().numpy() for x in loss_history]
    plt.plot(loss_history)
    plt.show()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model_max, test_loader, data_o, data_a, args)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

    with open(args.out_file, 'a') as f:
        f.write('{0}\t{1}\t{2}\t{7}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\n'.format(
            args.in_file[5:8], args.seed, args.aggregator, loss_test.item(), auroc_test, prc_test, f1_test,
            args.feature_type))


def test(model, loader, data_o, data_a, args):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    lbl = data_a.y

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            if args.cuda:
                label = label.cuda()

            pred, ind, k, output, loss2 = model(data_o.x, data_a.y, data_o.edge_index, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.float())
            loss = loss1 + 0.2 * loss2

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss
