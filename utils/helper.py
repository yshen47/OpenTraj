import json
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


def save_checkpoint(state, prefix=''):
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if 'tar' not in prefix:
                print("save a new best model")
                torch.save(state, prefix + '/model_best.pth.tar')
            else:
                print("save a checkpoint")
                torch.save(state, prefix)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        if not tries:
            raise error


def load_checkpoint(model, last_checkpoint):
    if last_checkpoint and os.path.isfile(last_checkpoint):
        checkpoint = torch.load(last_checkpoint)
        best_val_loss = checkpoint['valid_loss']
        steps = checkpoint["steps"]
        model.load_state_dict(checkpoint['model'])
        # Eiters is used to show logs as the continuation of another
        # training
        print("=> loaded checkpoint '{}' (best_val_loss {}))"
              .format(last_checkpoint, best_val_loss))
        return steps, best_val_loss
    else:
        print("=> no checkpoint found at '{}'".format(last_checkpoint))


def write_log(data, logfile):
    with open(logfile, 'w') as outfile:
        json.dump(data, outfile)


def normalize_nonsym_adj(adj):
    degree = np.asarray(adj.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv_sqrt = 1. / np.sqrt(degree)
    degree_inv_sqrt_mat = sp.diags([degree_inv_sqrt], [0])

    degree_inv = degree_inv_sqrt_mat.dot(degree_inv_sqrt_mat)

    adj_norm = degree_inv.dot(adj)

    return adj_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def compute_auc(preds, labels):
    return roc_auc_score(labels.astype(int), preds)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def compute_ap(gts, preds):
    aps = []
    for i in range(preds.shape[1]):
      ap, prec, rec = calc_pr(gts == i, preds[:,i:i+1,:,:])
      aps.append(ap)
    return aps

def calc_pr(gt, out, wt=None):
    gt = gt.astype(np.float64).reshape((-1,1))
    out = out.astype(np.float64).reshape((-1,1))

    tog = np.concatenate([gt, out], axis=1)*1.
    ind = np.argsort(tog[:,1], axis=0)[::-1]
    tog = tog[ind,:]
    cumsumsortgt = np.cumsum(tog[:,0])
    cumsumsortwt = np.cumsum(tog[:,0]-tog[:,0]+1)
    prec = cumsumsortgt / cumsumsortwt
    rec = cumsumsortgt / np.sum(tog[:,0])
    ap = voc_ap(rec, prec)
    return ap, rec, prec

def voc_ap(rec, prec):
    rec = rec.reshape((-1,1))
    prec = prec.reshape((-1,1))
    z = np.zeros((1,1))
    o = np.ones((1,1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    I = np.where(mrec[1:] != mrec[0:-1])[0]+1
    ap = np.sum((mrec[I] - mrec[I-1])*mpre[I])
    return ap

def plot_results(counts, ious, aps, classes, file_name):
    fig, ax = plt.subplots(1,1)
    conf = counts / 1.0 / np.sum(counts, 1, keepdims=True)
    conf = np.concatenate([conf, np.array(aps).reshape(-1,1),
                           np.array(ious).reshape(-1,1)], 1)
    conf = conf * 100.
    sns.heatmap(conf, annot=True, ax=ax, fmt='3.0f')
    arts = []
    # labels, title and ticks
    _ = ax.set_xlabel('Predicted labels')
    arts.append(_)
    _ = ax.set_ylabel('True labels')
    arts.append(_)
    _ = ax.set_title('Confusion Matrix, mAP: {:5.1f}, mIoU: {:5.1f}'.format(
      np.mean(aps)*100., np.mean(ious)*100.))
    arts.append(_)
    _ = ax.xaxis.set_ticklabels(classes + ['AP', 'IoU'], rotation=90)
    arts.append(_)
    _ = ax.yaxis.set_ticklabels(classes, rotation=0)
    arts.append(_)
    fig.savefig(file_name, bbox_inches='tight')


def compute_confusion_matrix(gts, preds):
    preds_cls = np.argmax(preds, 1)
    gts = gts[:,0,:,:]
    conf = confusion_matrix(gts.ravel(), preds_cls.ravel(), labels=np.arange(13))
    inter = np.diag(conf)
    union = np.sum(conf, 0) + np.sum(conf, 1) - np.diag(conf)
    union = np.maximum(union, 1) * 1.0
    return inter / union, conf

def draw_traj(model_dir, index, preds, inputs, background_img_dir=os.path.join('datasets', "ETH", "seq_eth", 'map.png')):
    canvass = np.zeros([250,250,3])
    canvass[:,:,:] = [255,255,255]
    # plot ground-truth points
    red = [0, 255, 255]
    green = [255, 0, 0]
    # for batch in inputs:
    #     for agent_traj in batch:
    #         for point in agent_traj:
    #             if point[0] != 0 and point[1] != 0:
    #                 x = int(point[0] * 10) + 75
    #                 y = int(point[1] * 10) + 32
    #                 canvass[x,y] = green
    for batch in preds:
        for agent_traj in batch:
            for point in agent_traj:
                x = int(point[0] * 10) + 75
                y = int(point[1] * 10) + 32
                canvass[x,y] = red

    if not os.path.exists(os.path.join(model_dir, 'visualization')):
        os.makedirs(os.path.join(model_dir, 'visualization'))
    if preds.sum() > 0:
        print(preds)
        cv2.imwrite(os.path.join(model_dir, 'visualization', str(index) + ".jpg"), canvass)