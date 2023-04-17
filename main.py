"""
GraphRec + FBNE
"""

import os
import time
import argparse
import pickle
from math import sqrt
import numpy as np
import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, label_ranking_average_precision_score
from tqdm import tqdm

from torchmetrics.classification import MultilabelRankingAveragePrecision

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator

import matplotlib.pyplot as plt
import time
from collections import defaultdict

from utils import collate_fn
from model import GraphRec
from modelOriginal import GraphRecOriginal
from dataloader import GRDataset

import torch.nn.functional as F
from Folded_Encoders import Folded_Encoder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default= os.getcwd() + "/datasets/", help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--dataset_type', default='Linkedin', help='input batch size')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
#8, 16, 32, 64, 128
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='N', help='input batch size for testing')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(bipartite):
    print('Loading data...')
    if not bipartite:
        with open(args.dataset_path + args.dataset_type + '/dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)

        with open(args.dataset_path + args.dataset_type + '/list.pkl', 'rb') as f:
            u_items_list = pickle.load(f)
            u_users_list = pickle.load(f)
            u_users_items_list = pickle.load(f)
            i_users_list = pickle.load(f)
            (user_count, item_count, rate_count) = pickle.load(f)

        train_data = GRDataset(train_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
        valid_data = GRDataset(valid_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
        test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
        valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
        test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

        model = GraphRec(user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)

        if args.test:
            print('Load checkpoint and testing...')
            ckpt = torch.load('best_checkpoint.pth.tar')
            model.load_state_dict(ckpt['state_dict'])
            mae, rmse = validate(test_loader, model)
            print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
            return

        optimizer = optim.RMSprop(model.parameters(), args.lr)
        criterion = nn.MSELoss()
        scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    else:
        with open(args.dataset_path + args.dataset_type + '/dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)

        with open(args.dataset_path + args.dataset_type + '/list.pkl', 'rb') as f:
            history_u_lists = pickle.load(f)
            history_ur_lists = pickle.load(f)
            history_v_lists = pickle.load(f)
            history_vr_lists = pickle.load(f)
            walks_u = pickle.load(f)
            walks_v = pickle.load(f)
            train_u = pickle.load(f)
            train_v = pickle.load(f)
            train_r = pickle.load(f)
            test_u = pickle.load(f)
            test_v = pickle.load(f)
            test_r = pickle.load(f)
            valid_u = pickle.load(f)
            valid_v = pickle.load(f)
            valid_r = pickle.load(f)
            social_adj_lists = pickle.load(f)
            ratings_list = pickle.load(f)

        embed_dim = args.embed_dim

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                                  torch.FloatTensor(train_r))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                                 torch.FloatTensor(test_r))
        validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                                 torch.FloatTensor(valid_r))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        num_users = history_u_lists.__len__()
        num_items = history_v_lists.__len__()
        num_ratings = ratings_list.__len__()
        print("number of users, items, ratings: ", (num_users, num_items, num_ratings))

        u2e = nn.Embedding(num_users, embed_dim).to(device)
        v2e = nn.Embedding(num_items, embed_dim).to(device)
        r2e = nn.Embedding(num_ratings, embed_dim).to(device)
        # user feature
        agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
        enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device,
                                   uv=True)
        enc_u = Folded_Encoder(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, 5, walks_u,
                               base_model=enc_u_history, cuda=device)

        # item feature: user * rating
        agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
        enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device,
                                   uv=False)

        enc_v = Folded_Encoder(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, 5, walks_v,
                               base_model=enc_v_history, cuda=device)

        model = GraphRecOriginal(enc_u, enc_v, r2e).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
        criterion = nn.MSELoss()
        scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    start = time.time()

    epoch_rmse = defaultdict(list)
    epoch_mae = defaultdict(list)
    epoch_mrr = defaultdict(list)

    rmse = 9999.0
    mae = 9999.0
    endure_count = 0

    best_mae = 9999.0
    best_rmse = 9999.0

    total_accuracy = defaultdict(list)
    total_predictions = defaultdict(list)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)

        if not bipartite:
            train(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)
            mae, rmse = validate(valid_loader, model)


            # store best loss and save a model checkpoint
            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

            if epoch == 0:
                best_mae = mae
            elif mae < best_mae:
                best_mae = mae

                epoch_rmse[epoch].append(rmse)
                epoch_mae[epoch].append(mae)

                torch.save(ckpt_dict, 'best_checkpoint.pth.tar')


            print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}'.format(epoch, mae, rmse, best_mae))

        else:
            trainBipartite(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
            # expected_rmse, mae , mrr = testBipartite(model, device, test_loader)
            # expected_rmse, mae, mrr = testBipartite(model, device, test_loader, total_accuracy, total_predictions, epoch)
            expected_rmse, mae, mrr = testBipartite(model, device, test_loader)

            if epoch == 0:
                pass
            else:
                epoch_rmse[epoch].append(expected_rmse)
                epoch_mae[epoch].append(mae)
                epoch_mrr[epoch].append(mrr)

            if best_rmse > expected_rmse:
                best_rmse = expected_rmse
                best_mae = mae
                endure_count = 0
            else:
                endure_count += 1

            print("rmse: %.4f, mae: %.4f , mrr: %.4f" % (expected_rmse, mae, mrr))

            if endure_count > 10:
                break


    rmse_list = sorted(epoch_rmse.items())
    X_rmse, y_rmse = zip(*rmse_list)

    plt.title("RMSE vs Epoch - " + args.dataset_type + " - " + str(args.test_batch_size) + " batch size")
    plt.plot(list(map(int, X_rmse)), y_rmse, marker='o', markerfacecolor='r')
    plt.show()

    mrr_list = sorted(epoch_mrr.items())
    X_mrr, y_mrr = zip(*mrr_list)

    plt.title("MRR vs Epoch - " + args.dataset_type + " - " + str(args.test_batch_size) + " batch size")
    plt.plot(list(map(int, X_mrr)), y_mrr, marker='o', markerfacecolor='r')
    plt.show()

    mae_list = sorted(epoch_mae.items())
    X_mae, y_mae = zip(*mae_list)

    plt.title("MAE vs Epoch - " + args.dataset_type + " - " + str(args.test_batch_size) + " batch size")
    plt.plot(list(map(int, X_mae)), y_mae, marker='o', markerfacecolor='r')
    plt.show()

    end = time.time()
    print("Time Elapsed: ", (end - start) / 60)

def trainBipartite(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('Training: [%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0

def testBipartite(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            # val_output = torch.round(model.forward(test_u, test_v))
            val_output = model.forward(test_u, test_v)
            # val_output = torch.clamp(val_output, min=0, max=1)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))

    tmp_pred = normalize(tmp_pred, (0,4), (0,1))
    tmp_pred = round_to_nearest_quarter(tmp_pred)
    target = normalize(target, (0,4), (0,1))

    # metric = MultilabelRankingAveragePrecision(num_labels=len(target))
    # mrr = metric(tmp_pred, target)

    mrr = label_ranking_average_precision_score(target, tmp_pred)
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae, mrr

def label_ranking_average_precision_score(y_true, y_pred):

    # Ensure input is a numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Initialize variables
    num_samples = y_true.shape[0]
    num_labels = y_pred.shape[0]
    sum_precision = 0
    # Initialize array to store average precision scores for each label
    avg_precision_scores = np.zeros(num_labels)

    # Iterate through each label
    for label in range(num_labels):
        # Initialize array to store precision scores for each sample
        precision_scores = np.zeros(num_samples)

        # Initialize variable to store total number of correct predictions for current label
        num_correct = 0

        # Iterate through each sample
        for i in range(num_samples):
            # If current label is ranked correctly in current sample, increment num_correct and calculate precision score
            if y_true[i] == y_pred[i]:
                num_correct += 1
                precision_scores[i] = num_correct / (label + 1)
            else:
                precision_scores[i] = num_correct / (label + 1)

        # Calculate average precision score for current label
        avg_precision_scores[label] = np.mean(precision_scores)

    # Calculate overall label ranking average precision score
        overall_score = np.mean(avg_precision_scores)

    return overall_score

def train(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (uids, iids, labels, u_items, u_users, u_users_items, i_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        
        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, u_users, u_users_items, i_users)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(uids) / (time.time() - start)))

        start = time.time()

def validate(valid_loader, model):
    model.eval()
    errors = []
    with torch.no_grad():
        for uids, iids, labels, u_items, u_users, u_users_items, i_users in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            u_items = u_items.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)
            preds = model(uids, iids, u_items, u_users, u_users_items, i_users)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse

def normalize(values, actual_bounds, desired_bounds):
    return [desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]) for x in values]

def round_to_nearest_quarter(values):
    return [round(x * 4)/4 for x in values]

if __name__ == '__main__':
    bipartite = True
    main(bipartite)
