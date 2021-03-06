#!/usr/bin/env python
# encoding: utf-8

import numpy as np, sys, math, os, h5py
from sklearn.metrics import roc_curve, precision_recall_curve, auc, r2_score
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from multiprocessing import Pool
from scipy import stats
# from bayes_opt import BayesianOptimization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate ids for k-flods cross-validation
def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = []; test_ids = []; valid_ids = []
    if k_folds == 1:
       train_num = int(seqs_num*0.7)
       test_num = seqs_num - train_num
       valid_num = int(train_num*ratio)
       train_num = train_num - valid_num
       index = list(range(seqs_num))
       train_ids.append(np.asarray(index[:train_num]))
       valid_ids.append(np.asarray(index[train_num:train_num+valid_num]))
       test_ids.append(np.asarray(index[train_num+valid_num:]))
    else:
       each_fold_num = int(math.ceil(seqs_num/k_folds))
       for fold in range(k_folds):
           index = list(range(seqs_num))
           index_slice = index[fold*each_fold_num:(fold+1)*each_fold_num]
           index_left = list(set(index) - set(index_slice))
           test_ids.append(np.asarray(index_slice))
           train_num = len(index_left) - int(len(index_left) * ratio)
           train_ids.append(np.asarray(index_left[:train_num]))
           valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)

# Compute the roc AUC and the precision-recall AUC
def ComputeAUC(y_pred, y_real):
    # roc_auc_score(y_real, y_pred)
    fpr, tpr, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)
    # average_precision_score(y_real, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_real,y_pred)
    pr_auc = auc(recall,precision)

    return (roc_auc, pr_auc)


#Compute the pearson corelation coefficient
def ComputePCC(y_pred, y_real):
    # pearson coefficient
    coeff, pvalue = stats.pearsonr(y_pred, y_real)
    score = r2_score(y_real, y_pred)
    return coeff, score

# Generate random hyper-paramter settings
def RandomSample():
    space = {
    'DROPOUT': hp.choice( 'drop', ( 0.2, 0.5, 0.7)),
    'DROPOUT2': hp.choice('drop2', (0.2, 0.5)),
    'DROPOUT1': hp.choice('drop1', (0.2, 0.5)),
    }
    params = sample(space)
    return params


# select the best paramter setting
def SelectBest(history_all, file_path, fold, monitor='val_loss'):
    if monitor == 'val_loss':
       loss = 100000.
       for num, History in list(history_all.items()):
           if np.min(History.history['val_loss']) < loss:
              best_num = int(num)
              loss = np.min(History.history['val_loss'])
    else:
       acc = 0.
       for num, History in list(history_all.items()):
           if np.max(History.history['val_acc']) > acc:
              best_num = int(num)
              acc = np.max(History.history['val_acc'])

    del_num = list(range(len(history_all)))
    del_num.pop(best_num)
    # delete the useless model paramters
    for num in del_num:
       os.remove(file_path + 'params%d_bestmodel_%dfold.hdf5' %(num, fold))

    return best_num

# plot and save the training process
def PlotandSave(History, filepath, fold, monitor='val_loss'):
    if monitor == 'val_loss':
       train_loss = History.history['loss']
       valid_loss = History.history['val_loss']
       x = list(range(len(train_loss)))

       plt.figure(num = fold)
       plt.title('mode loss')
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.plot(x, train_loss, 'r-', x, valid_loss, 'g-')
       plt.legend(['train_loss', 'valid_loss'], loc = 'upper left')
       #plt.show()
    else:
       train_acc = History.history['acc']
       valid_acc = History.history['val_acc']
       x = list(range(len(train_acc)))

       plt.figure(num = fold)
       plt.title('model accuracy')
       plt.ylabel('accuracy')
       plt.xlabel('epoch')
       plt.plot(x, train_acc, 'r-', x, valid_acc, 'g-')
       plt.legend(['train_acc', 'valid_acc'], loc = 'upper left')
       #plt.show()

    plt.savefig(filepath, format = 'png')



