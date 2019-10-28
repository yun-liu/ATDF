#!/usr/bin/python

import os
import numpy as np
import multiprocessing as mp


def handle_func(img_files, res_files, out_files, start_id, end_id):
    for idx in range(start_id, end_id):
        print img_files[idx]
        order = 'python dense_hsal.py ' + img_files[idx] + ' ' \
            + res_files[idx] + ' ' + out_files[idx]
        os.system(order)


with open('../../saliency_data/SOD.lst') as f:
    test_lst = f.readlines()
iids = [x.strip() for x in test_lst]
img_files = ['../../saliency_data/'+x for x in iids]
res_files = ['../../saliency_data/'+x[:-4]+'_vgg.png' for x in iids]
out_files = ['../../saliency_data/'+x[:-4]+'_vgg_crf.png' for x in iids]

work_nums = 28
sub_process = []
for i in range(work_nums):
    start_id = len(img_files) / work_nums * i
    if i == work_nums - 1:
        end_id = len(img_files)
    else:
        end_id = len(img_files) / work_nums * (i + 1)
    process = mp.Process(target=handle_func, args=(img_files, res_files,
        out_files, start_id, end_id,))
    sub_process.append(process)
for p in sub_process:
    p.start()
for p in sub_process:
    p.join()
