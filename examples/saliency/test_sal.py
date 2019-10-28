import os
import cv2
import time
import numpy as np
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


data_root = '../../saliency_data/'
with open('../../saliency_data/SOD.lst') as f:
    test_lst = f.readlines()
test_lst = [data_root+x.strip() for x in test_lst]

# remove the following two lines if testing with cpu
caffe.set_mode_gpu()
# choose which GPU you want to use
caffe.set_device(0)
# load net
net = caffe.Net('test_vgg16.prototxt', 'snapshots/atdf_sal_vgg16_iter_20000.caffemodel', caffe.TEST)

start_time = time.time()
for idx in range(len(test_lst)):
    print test_lst[idx]
    # load image
    im = Image.open(test_lst[idx])
    w, h = im.size
    if w*h > 290000:
        ratio = np.sqrt(290000.0/(w*h))
        im = im.resize((int(w*ratio), int(h*ratio)), Image.ANTIALIAS)
    im = np.array(im, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    net.forward()

    dsn1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    dsn2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    dsn3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    dsn4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    dsn5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    dsn6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    res = fuse
    EPSILON = np.finfo(np.float32).eps
    res = (res - np.min(res)) / (np.max(res) - np.min(res) + EPSILON)
    res = (res*255).astype(np.uint8)
    if w*h > 290000:
        res = cv2.resize(res, (w, h), cv2.INTER_LINEAR)
    cv2.imwrite(test_lst[idx][:-4] + '_vgg.png', res)
diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))
