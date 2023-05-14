import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

from torchmetrics.classification import MultilabelRankingAveragePrecision

import torch
# y_true = np.array([[2.,3.,0.]])
# y_score = np.array([[2.,3.,0.]])
# print(y_true)
# print(type(y_true.shape))
# print(y_score.shape)
# print(label_ranking_average_precision_score(y_true, y_score))

# y_true = np.array([[2.,3.,0.]])
# y_score = np.array([[2.,3.,0.]])
y_true = np.array([[1.8609242,1.9057674,3.0060685]])
y_score = np.array([[2.,2.,4.]])
print(label_ranking_average_precision_score(y_true, y_score))
# metric = MultilabelRankingAveragePrecision(num_labels=3)
#
# print(preds)
# print(target)
# print(metric(preds, target))