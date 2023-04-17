import argparse
import random
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl.function as fn
from itertools import combinations
import os
import pickle
from dataloader import GRDataset
from torch.utils.data import DataLoader
from modelOriginal import GraphRecOriginal
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Folded_Encoders import Folded_Encoder
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []

# def train_regression(model, features, labels_local, train_idx, epochs, weight_decay, lr):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     loss_fcn = torch.nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         model.train()
#         logits = model(features)
#         loss = loss_fcn(logits[train_idx], labels_local[train_idx])
#         optimizer.zero_grad()
#         loss.backward()
#     return model

def trainBipartite(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        # if i % 100 == 0:
        #     print('Training: [%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
        #         epoch, i, running_loss / 100, best_rmse, best_mae))
        #     running_loss = 0.0
    return model

def test_regression(model, test_features, test_labels, idx_test):
    model.eval()
    return evaluate(model, test_features, test_labels, idx_test)

def main():
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

    train_shot = args.train_shot
    test_shot = args.test_shot
    # data = load_data(args)
    # features = torch.FloatTensor(data.features)
    # labels = torch.LongTensor(data.labels)
    # train_mask = torch.ByteTensor(data.train_mask)
    # val_mask = torch.ByteTensor(data.val_mask)
    # test_mask = torch.ByteTensor(data.test_mask)
    # in_feats = features.shape[1]
    n_classes = 2

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


    # g = data._g
    step = 50
    total_accuracy_meta_test = []
    accuracy_meta_test = []
    node_num = num_users
    iteration = 10
    class_label = list(range(0, num_items))
    combination = list(combinations(class_label, 3))

    print(len(combination))
    for i in range(len(combination)):
        print('Cross_Validation: ',i+1)
        test_label = list(combination[i])
        train_label = [n for n in class_label if n not in test_label]
        print('Cross_Validation {} Train_Label_List {}: '.format(i + 1, train_label))
        print('Cross_Validation {} Test_Label_List {}: '.format(i + 1, test_label))
        model = GraphRecOriginal(enc_u, enc_v, r2e).to(device)

        for j in range(iteration):
            labels_local = list(history_v_lists.values())[:]
            select_class = random.sample(train_label, n_classes)
            print('EPOCH {} ITERATION {} Train_Label: {}'.format(i+1, j+1, select_class))
            class1_idx = []
            class2_idx = []
            for k in range(node_num):
                if(labels_local[k] == select_class[0]):
                    class1_idx.append(k)
                    labels_local[k] = 0
                elif(labels_local[k] == select_class[1]):
                    class2_idx.append(k)
                    labels_local[k] = 1


            for epoch in tqdm(range(args.epoch)):
                class1_train = random.sample(class1_idx, train_shot)
                class2_train = random.sample(class2_idx, train_shot)
                class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                train_idx = class1_train + class2_train
                random.shuffle(train_idx)
                test_idx = class1_test + class2_test
                random.shuffle(test_idx)
                # model = trainBipartite(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
                model = trainBipartite(model, device, train_loader, optimizer, epoch)
                acc_query = test_regression(model, num_users, labels_local, test_idx)
                print("acc_query: ", acc_query)
                accuracy_meta_test.append(acc_query)
                reset_array()
        print('Cross_Validation: {} Meta-Train_Accuracy: {}'.format(i + 1, torch.tensor(accuracy_meta_test).numpy().mean()))
        accuracy_meta_test = []
        torch.save(model.state_dict(), 'model.pkl')

        labels_local = num_items.clone().detach()
        select_class = random.sample(test_label, 2)
        print('EPOCH {} Test_Label {}: '.format(i + 1, select_class))
        class1_idx = []
        class2_idx = []
        reset_array()
        for k in range(node_num):
            if (labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif (labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1
        for m in range(step):
            class1_train = random.sample(class1_idx, test_shot)
            class2_train = random.sample(class2_idx, test_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)

            model_meta_trained = model = GraphRecOriginal(enc_u, enc_v, r2e).to(device)

            model_meta_trained.load_state_dict(torch.load('model.pkl'))

            model = trainBipartite(model, device, train_loader, optimizer, epoch)
            acc_test = test_regression(model_meta_trained, num_users, labels_local, test_idx)
            accuracy_meta_test.append(acc_test)
            total_accuracy_meta_test.append(acc_test)
            reset_array()
        # if args.dataset == 'cora':
        #     with open('pool_cora.txt', 'a') as f:
        #         f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i + 1, torch.tensor(
        #             accuracy_meta_test).numpy().mean()))
        #         f.write('\n')
        # elif args.dataset == 'citeseer':
        #     with open('pool_citeseer.txt', 'a') as f:
        #         f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i + 1, torch.tensor(
        #             accuracy_meta_test).numpy().mean()))
        #         f.write('\n')
        accuracy_meta_test = []

    # if args.dataset == 'cora':
    #     with open('pool_cora.txt', 'a') as f:
    #         f.write('Dataset: {}, Train_Shot: {}, Test_Shot: {}'.format(args.dataset, train_shot, test_shot))
    #         f.write('\n')
    #         f.write('Total_Meta-Test_Accuracy: {}'.format(torch.tensor(total_accuracy_meta_test).numpy().mean()))
    #         f.write('\n')
    #         f.write('\n\n\n')
    # elif args.dataset == 'citeseer':
    #     with open('pool_citeseer.txt', 'a') as f:
    #         f.write('Dataset: {}, Train_Shot: {}, Test_Shot: {}'.format(args.dataset, train_shot, test_shot))
    #         f.write('\n')
    #         f.write('Total_Meta-Test_Accuracy: {}'.format(torch.tensor(total_accuracy_meta_test).numpy().mean()))
    #         f.write('\n')
    #         f.write('\n\n\n')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='GCN')
#     register_data_args(parser)
#     parser.add_argument("--dropout", type=float, default=0.5,
#                         help="dropout probability")
#     # parser.add_argument("--dataset", type=str, default="cora")
#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2,
#                         help="learning rate")
#     parser.add_argument("--weight_decay", type=float, default=5e-4,
#                         help="Weight for L2 loss")
#     parser.add_argument('--train_shot', type=int, default=20, help='How many shot during meta-train')
#     parser.add_argument('--test_shot', type=int, default=1, help='How many shot during meta-test')
#     args = parser.parse_args()
#     print(args)
#
#     main(args)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default= os.getcwd() + "/datasets/", help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--dataset_type', default='Linkedin', help='input batch size')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
#8, 16, 32, 64, 128
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='N', help='input batch size for testing')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--train_shot', type=int, default=20, help='How many shot during meta-train')
parser.add_argument('--test_shot', type=int, default=1, help='How many shot during meta-test')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # bipartite = True
    # main(bipartite)
    main()