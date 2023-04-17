# -*- coding: utf-8 -*-
import itertools
import random
import pickle
import argparse
import re
import unicodedata
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from random import shuffle

import graph

from networkx.algorithms import bipartite as bi

import os

random.seed(1234)

workdir = os.getcwd() + '/datasets/'

bipartite = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Linkedin', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()


# load data

def save_homogenous_graph_to_file(A, datafile, index_row, index_item):
    (M, N) = A.shape  # 8371 * 8371
    csr_dict = A.__dict__
    data = csr_dict.get("data")
    indptr = csr_dict.get("indptr")
    indices = csr_dict.get("indices")
    col_index = 0
    with open(datafile, 'w') as fw:
        for row in range(M):
            for col in range(indptr[row], indptr[row + 1]):
                r = row
                c = indices[col]
                fw.write(str(index_row.get(r)) + "\t" + str(index_item.get(c)) + "\t" + str(data[col_index]) + "\n")
                col_index += 1

def calculate_centrality(G, uSet, bSet, mode='hits'):
    authority_u = {}
    authority_v = {}
    if mode == 'degree_centrality':
        a = nx.degree_centrality(G)
    else:
        # h, a = nx.hits(G)
        a = nx.pagerank(G)

    max_a_u, min_a_u, max_a_v, min_a_v = 0, 100000, 0, 100000

    for node in G.nodes():
        if node in uSet:
            if max_a_u < a[node]:
                max_a_u = a[node]
            if min_a_u > a[node]:
                min_a_u = a[node]
        if node in bSet:
            if max_a_v < a[node]:
                max_a_v = a[node]
            if min_a_v > a[node]:
                min_a_v = a[node]

    for node in G.nodes():
        if node in uSet:
            if max_a_u - min_a_u != 0:
                authority_u[node] = (float(a[node]) - min_a_u) / (max_a_u - min_a_u)
            else:
                authority_u[node] = 0
        if node in bSet:
            if max_a_v - min_a_v != 0:
                authority_v[node] = (float(a[node]) - min_a_v) / (max_a_v - min_a_v)
            else:
                authority_v[node] = 0
    return authority_u, authority_v

def get_random_walks_restart(datafile, hits_dict, percentage):
    G = graph.load_edgelist(datafile, undirected=True)
    print("Folded ==> number of nodes: {}".format(len(G.nodes())))
    print("Deepwalk process")
    # walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
    walks = graph.build_deepwalk_corpus(G, hits_dict, percentage, alpha=0, rand=random.Random())
    print("Deepwalk process done..")
    return G, walks

def generate_bipartite_folded_walks(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu):
    BiG = nx.Graph()
    node_u = history_u_lists.keys()
    node_v = history_v_lists.keys()
    node_u = sorted(node_u)
    node_v = sorted(node_v)

    BiG.add_nodes_from(node_u, bipartite=0)
    BiG.add_nodes_from(node_v, bipartite=1)
    BiG.add_weighted_edges_from(edge_list_uv + edge_list_vu)
    A = bi.biadjacency_matrix(BiG, node_u, node_v, dtype=np.float, weight='weight', format='csr')

    row_index = dict(zip(node_u, itertools.count()))  # node_u_id_original : index_new
    col_index = dict(zip(node_v, itertools.count()))  # node_v_id_original : index_new

    index_row = dict(zip(row_index.values(), row_index.keys()))  # index_new : node_u_id_original
    index_item = dict(zip(col_index.values(), col_index.keys()))

    AT = A.transpose()
    fw_u = os.path.join(path, "homogeneous_u.dat")
    fw_v = os.path.join(path, "homogeneous_v.dat")
    save_homogenous_graph_to_file(A.dot(AT), fw_u, index_row, index_row)
    save_homogenous_graph_to_file(AT.dot(A), fw_v, index_item, index_item)

    authority_u, authority_v = calculate_centrality(BiG, node_u, node_v)  # todo task

    G_u, walks_u = get_random_walks_restart(fw_u, authority_u, percentage=0.30)
    G_v, walks_v = get_random_walks_restart(fw_v, authority_v, percentage=0.30)

    return G_u, walks_u, G_v, walks_v

def normalize_job_title(title):
    title = unicodedata.normalize('NFKC', title)
    title = re.sub(r'【.*】', '', title)
    title = re.sub(r'\[.*\]', '', title)
    title = re.sub(r'「.*」', '', title)
    title = re.sub(r'\(.*\)', '', title)
    title = re.sub(r'\<.*\>', '', title)
    title = re.sub(r'[※@◎].*$', '', title)
    return title.lower()

def preprocess():
    path = os.getcwd()

    if not bipartite:
        if args.dataset == 'Ciao':
            click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
            trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
        elif args.dataset == 'Epinions':
            click_f = np.loadtxt(workdir + 'Epinions/ratings_data.txt', dtype=np.int32)
            trust_f = np.loadtxt(workdir + 'Epinions/trust_data.txt', dtype=np.int32)
        elif args.dataset == 'Linkedin':
            click_f = np.loadtxt(workdir + 'Linkedin/updated_fixed_linkedin_u2s.txt', dtype=np.int32)
            trust_f = np.loadtxt(workdir + 'Linkedin/updated_fixed_linkedin_u2pos.txt', dtype=np.int32)

        click_list = []
        trust_list = []

        u_items_list = []
        u_users_list = []
        u_users_items_list = []
        i_users_list = []

        user_count = 0
        item_count = 0
        rate_count = 0

        for s in click_f:
            uid = s[0]
            iid = s[1]
            if args.dataset == 'Ciao':
                label = s[3]
            elif args.dataset == 'Epinions':
                label = s[2]
            elif args.dataset == 'Linkedin':
                label = s[2]

            if uid > user_count:
                user_count = uid
            if iid > item_count:
                item_count = iid
            if label > rate_count:
                rate_count = label
            click_list.append([uid, iid, label])

        pos_list = []
        for i in range(len(click_list)):
            pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

        pos_list = list(set(pos_list))

        # train, valid and test data split
        random.shuffle(pos_list)
        num_test = int(len(pos_list) * args.test_prop)
        test_set = pos_list[:num_test]
        valid_set = pos_list[num_test:2 * num_test]
        train_set = pos_list[2 * num_test:]
        print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set),
                                                                              len(test_set)))

        with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])
        valid_df = pd.DataFrame(valid_set, columns=['uid', 'iid', 'label'])
        test_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])

        click_df = pd.DataFrame(click_list, columns=['uid', 'iid', 'label'])
        train_df = train_df.sort_values(axis=0, ascending=True, by='uid')

        """
		u_items_list
		"""
        for u in tqdm(range(user_count + 1)):
            hist = train_df[train_df['uid'] == u]
            u_items = hist['iid'].tolist()
            u_ratings = hist['label'].tolist()
            if u_items == []:
                u_items_list.append([(0, 0)])
            else:
                u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

        train_df = train_df.sort_values(axis=0, ascending=True, by='iid')

        """
		i_users_list
		"""
        for i in tqdm(range(item_count + 1)):
            hist = train_df[train_df['iid'] == i]
            i_users = hist['uid'].tolist()
            i_ratings = hist['label'].tolist()
            if i_users == []:
                i_users_list.append([(0, 0)])
            else:
                i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

        for s in trust_f:
            uid = s[0]
            fid = s[1]
            if uid > user_count or fid > user_count:
                continue
            trust_list.append([uid, fid])

        trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
        trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')

        """
		u_users_list
		u_users_items_list
		"""
        for u in tqdm(range(user_count + 1)):
            hist = trust_df[trust_df['uid'] == u]
            u_users = hist['fid'].unique().tolist()
            if u_users == []:
                u_users_list.append([0])
                u_users_items_list.append([[(0, 0)]])
            else:
                u_users_list.append(u_users)
                uu_items = []
                for uid in u_users:
                    uu_items.append(u_items_list[uid])
                u_users_items_list.append(uu_items)

        with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
            pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.dataset == 'Ciao':
            click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
            trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
        elif args.dataset == 'Epinions':
            click_f = np.loadtxt(workdir + 'Epinions/ratings_data.txt', dtype=np.int32)
            trust_f = np.loadtxt(workdir + 'Epinions/trust_data.txt', dtype=np.int32)
        elif args.dataset == 'Linkedin':
            click_f = np.loadtxt(workdir + 'Linkedin/updated_fixed_linkedin_u2exp.txt', dtype=np.int32)
            trust_f = np.loadtxt(workdir + 'Linkedin/updated_fixed_linkedin_u2pos.txt', dtype=np.int32)

        # click_list = []
        # trust_list = []
        #
        # u_items_list = []
        # u_users_list = []
        # u_users_items_list = []
        # i_users_list = []
        #
        user_count = 0
        item_count = 0
        rate_count = 0

        uSet_u2u = set()
        uSet_u2b = set()
        bSet_u2b = set()

        social_adj_lists = defaultdict(set)
        history_u_lists = defaultdict(list)
        history_v_lists = defaultdict(list)

        history_ur_lists = defaultdict(list)
        history_vr_lists = defaultdict(list)

        neg_neighbors_u2u = defaultdict()
        neg_neighbors_u2b = defaultdict()

        total_users = set()
        total_positions = set()

        G = nx.Graph()
        G.name = args.dataset

        for s in click_f:
            uid = s[0]
            iid = s[1]
            if args.dataset == 'Ciao':
                label = s[3]
                uSet_u2b.add(uid)
                bSet_u2b.add(iid)
                G.add_edge(uid, iid, type='u2b', rating=label)
            elif args.dataset == 'Epinions':
                label = s[2]
                uSet_u2b.add(uid)
                bSet_u2b.add(iid)
                G.add_edge(uid, iid, type='u2b', rating=label)
            elif args.dataset == 'Linkedin':
                label = s[2]
                uSet_u2b.add(uid)
                bSet_u2b.add(iid)
                G.add_edge(uid, iid, type='u2b', rating=label)

            if uid > user_count:
                user_count = uid
            if iid > item_count:
                item_count = iid
            if label > rate_count:
                rate_count = label

        for s in trust_f:
            uid = s[0]
            fid = s[1]
            if uid > user_count or fid > user_count:
                continue
            else:
                uSet_u2u.add(uid)
                uSet_u2u.add(fid)
                G.add_edge(uid, fid, type='u2u')

        print(nx.info(G))

        node_names = nx.get_node_attributes(G, 'name')  # key-value dict {'id':'name'}
        inv_map = {v: k for k, v in node_names.items()}

        # uSet_u2u = set([inv_map.get(name) for name in uSet_u2u])
        # uSet_u2b = set([inv_map.get(name) for name in uSet_u2b])
        # bSet_u2b = set([inv_map.get(name) for name in bSet_u2b])

        edge_list_uv = []
        edge_list_vu = []

        for node in G:
            for nbr in G[node]:
                if G[node][nbr]['type'] == 'u2u':
                    # print("social_adj_lists[node]: ", social_adj_lists[node])
                    # print("nbr: ", nbr)
                    social_adj_lists[node].add(nbr)
                if G[node][nbr]['type'] == 'u2b':
                    r = G[node][nbr]['rating'] - 1
                    if node in uSet_u2b and nbr in bSet_u2b:
                        # print("node: ", node)
                        history_u_lists[node].append(nbr)
                        history_v_lists[nbr].append(node)
                        history_ur_lists[node].append(r)
                        history_vr_lists[nbr].append(r)
                        edge_list_uv.append((node, nbr, r))
                        edge_list_vu.append((nbr, node, r))
                    if nbr in uSet_u2b and node in bSet_u2b:
                        history_u_lists[nbr].append(node)
                        history_v_lists[node].append(nbr)
                        history_ur_lists[nbr].append(r)
                        history_vr_lists[node].append(r)
                        edge_list_uv.append((nbr, node, r))
                        edge_list_vu.append((node, nbr, r))

        print("length of history u: ", len(history_u_lists))
        print("length of history v: ", len(history_v_lists))

        G_u, walks_u, G_v, walks_v = generate_bipartite_folded_walks(path, history_u_lists, history_v_lists,
                                                                     edge_list_uv,
                                                                     edge_list_vu)

        # train, valid and test data split
        data = []
        for (u, v) in G.edges():
            if G[u][v]['type'] == 'u2b':
                r = G[u][v]['rating'] - 1
                if u in uSet_u2b:
                    data.append((u, v, r))
                else:
                    data.append((v, u, r))

        size = len(data)
        # train_set = data[:int(0.8 * size)]  # 35704
        # test_set = data[int(0.8 * size):]  # 8927
        # valid_set = data[:len(test_set)//2]
        shuffle(data)

        train_set, valid_set, test_set = np.split(data, [int(len(data) * 0.8), int(len(data) * 0.9)])

        print(
            'Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set),
                                                                            len(test_set)))

        with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r = [], [], [], [], [], [], [], [], []

        for u, v, r in train_set:
            train_u.append(u)
            train_v.append(v)
            train_r.append(r)

        for u, v, r in test_set:
            test_u.append(u)
            test_v.append(v)
            test_r.append(r)

        for u, v, r in valid_set:
            valid_u.append(u)
            valid_v.append(v)
            valid_r.append(r)

        ratings_list = [0, 1, 2, 3, 4]

        # with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
        # 	pickle.dump(history_u_lists, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(history_ur_lists, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(history_v_lists, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(history_vr_lists, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(walks_u, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(walks_v, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(train_u, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(train_v, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(train_r, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(test_u, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(test_v, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(test_r, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(valid_u, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(valid_v, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(valid_r, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(social_adj_lists, f, pickle.HIGHEST_PROTOCOL)
        # 	pickle.dump(ratings_list, f, pickle.HIGHEST_PROTOCOL)

        # return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list

        # ------------------------------reindexed users and items respectively------------------------
        _social_adj_lists = defaultdict(set)
        _history_u_lists = defaultdict(list)
        _history_v_lists = defaultdict(list)

        _history_ur_lists = defaultdict(list)
        _history_vr_lists = defaultdict(list)
        _train_u, _train_v, _train_r, _test_u, _test_v, _test_r, _valid_u, _valid_v, _valid_r = [], [], [], [], [], [], [], [], []

        user_id_dic = {v: k for k, v in dict(enumerate(history_u_lists.keys())).items()}
        item_id_dic = {v: k for k, v in dict(enumerate(history_v_lists.keys())).items()}

        for u in history_u_lists:
            _history_u_lists[user_id_dic[u]] = [item_id_dic[v] for v in history_u_lists[u]]

        for v in history_v_lists:
            _history_v_lists[item_id_dic[v]] = [user_id_dic[u] for u in history_v_lists[v]]

        for u in history_ur_lists:
            _history_ur_lists[user_id_dic[u]] = history_ur_lists[u]

        for v in history_vr_lists:
            _history_vr_lists[item_id_dic[v]] = history_vr_lists[v]

        for u in social_adj_lists:
            _social_adj_lists[user_id_dic[u]] = [user_id_dic[us] for us in social_adj_lists[u]]

        for u, v, r in train_set:
            if u in user_id_dic.keys() and v in item_id_dic.keys():
                _train_u.append(user_id_dic[u])
                _train_v.append(item_id_dic[v])
                _train_r.append(r)

        for u, v, r in test_set:
            if u in user_id_dic.keys() and v in item_id_dic.keys():
                _test_u.append(user_id_dic[u])
                _test_v.append(item_id_dic[v])
                _test_r.append(r)

        for u, v, r in valid_set:
            if u in user_id_dic.keys() and v in item_id_dic.keys():
                _valid_u.append(user_id_dic[u])
                _valid_v.append(item_id_dic[v])
                _valid_r.append(r)

        # re-index walks_u and walks_v
        _walks_u = defaultdict(list)
        _walks_v = defaultdict(list)
        for u in walks_u:
            _walks_u[user_id_dic[u]] = [user_id_dic[us] for us in walks_u[u]]
        for v in walks_v:
            _walks_v[item_id_dic[v]] = [item_id_dic[vs] for vs in walks_v[v]]

        with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
            pickle.dump(_history_u_lists, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_history_ur_lists, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_history_v_lists, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_history_vr_lists, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_walks_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_walks_v, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_train_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_train_v, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_train_r, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_test_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_test_v, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_test_r, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_valid_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_valid_v, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_valid_r, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(_social_adj_lists, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(ratings_list, f, pickle.HIGHEST_PROTOCOL)


preprocessData = preprocess()
