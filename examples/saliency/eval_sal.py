#!/usr/bin/env python

import os
import cv2
import numpy as np
import os.path as osp

data_root = '../../saliency_data/'
with open('../../saliency_data/SOD.lst') as f:
    test_lst = f.readlines()
iids = [x.strip() for x in test_lst]
gt_files = [data_root+x[:-4]+'.png' for x in iids]
res_files = [data_root+x[:-4]+'_vgg_crf.png' for x in iids]
assert len(gt_files) == len(res_files), 'The number of GT and Res must be equal'

nthresh = 254
beta2 = 0.3
EPSILON = np.finfo(np.float).eps
thresh = np.linspace(1./(nthresh+1), 1.-1./(nthresh+1), nthresh)
recall = np.zeros((nthresh, len(gt_files)))
precision = np.zeros((nthresh, len(gt_files)))
mae = np.zeros(len(gt_files))

for idx in range(len(gt_files)):
    print gt_files[idx]
    gt = cv2.imread(gt_files[idx], 0)
    res = cv2.imread(res_files[idx], 0)
    gt = gt == 255
    res = res.astype(np.float) / 255
    for t in range(nthresh):
        bi_res = res > thresh[t]
        intersection = np.sum(np.logical_and(gt == bi_res, gt))
        recall[t, idx] = intersection * 1. / (np.sum(gt) + EPSILON)
        precision[t, idx] = intersection * 1. / (np.sum(bi_res) + EPSILON)
    mae[idx] = np.sum(np.fabs(gt - res)) * 1. / (gt.shape[0] * gt.shape[1])

recall = np.mean(recall, axis=1)
precision = np.mean(precision, axis=1)
F_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + EPSILON)
with open('../../saliency_data/ATDF_SOD.txt', 'w') as fid:
    for idx in range(nthresh):
        fid.write('{:10f} {:10f} {:10f} {:10f}\n'.format(thresh[idx],
            recall[idx], precision[idx], F_beta[idx]))
print 'F_mean=', np.mean(F_beta)
print 'F_beta =', np.max(F_beta)
print 'MAE =', np.mean(mae)
