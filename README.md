# ATDF
A Simple Saliency Detection Approach via Automatic Top-Down Feature Fusion (Submitted to Neurocomputing)

### Introduction

It is widely accepted that the top sides of convolutional neural networks (CNNs) convey high-level semantic features, and the bottom sides contai low-level details. Therefore, most of recent salient object detection methods aim at designing effective fusion strategies for side-output features. Although significant progress has been achieved in this direction, the network architectures become more and more complex, which will make the future improvement difficult and heavily engineered. Moreover, the manually designed fusion strategies would be sub-optimal due to the large search space of possible solutions. To address above problems, we propose an Automatic Top-Down Fusion (ATDF) method, in which the global information at the top sides are flowed into bottom sides to guide the learning of low layers. We design a novel valve module and add it at each side to control the coarse semantic information flowed into a specific bottom side. Through these valve modules, each bottom side at the top-down pathway is expected to receive necessary top information. We also design a generator to improve the prediction capability of fused deep features for saliency detection. We perform extensive experiments to demonstrate that ATDF is simple yet effective and thus opens a new path for saliency detection.

### Training RCF

1. Clone the RCF repository
    ```Shell
    git clone https://github.com/yun-liu/ATDF.git
    ```
    
2. Build Caffe.

3. Put the training data in the folder of `$ROOT_DIR/saliency_data` with a training list like

    ```
    DUTS-TR/ILSVRC2012_test_00000004.jpg DUTS-TR/ILSVRC2012_test_00000004.png
    DUTS-TR/ILSVRC2012_test_00000018.jpg DUTS-TR/ILSVRC2012_test_00000018.png
    DUTS-TR/ILSVRC2012_test_00000019.jpg DUTS-TR/ILSVRC2012_test_00000019.png
    ```
   Here, `$ROOT_DIR/saliency_data/*.jpg` is the path of training images, and `$ROOT_DIR/saliency_data/*.png` is the path of the saliency ground truth. Maybe you need to change the path to your own data.


4. Download the pretrained vgg16 model from [here](http://mftp.mmcheng.net/liuyun/rcf/model/5stage-vgg.caffemodel).

5. Start training process by running the following commands:

    ```Shell
    cd $ROOT_DIR/examples/saliency/
    ./train.sh
    ```

### Testing RCF

1. Download the datasets you need, and extract them to `$ROOT_DIR/saliency_data/` folder with a data list for each dataset like

    ```
    ECSSD/0674.jpg
    ECSSD/0097.jpg
    ECSSD/0747.jpg
    ```
   Here, $ROOT_DIR/saliency_data/*.jpg is the path of test images.

2. Change the paths for the data and model in `test_sal.py`, and start test process by running the following commands:

    ```Shell
    cd $ROOT_DIR/examples/saliency/
    python test_sal.py
    ```
    
3. Follow [this website](https://github.com/Andrew-Qibin/dss_crf) to install crf package, and perform CRF by running the following commands:
    
    ```Shell
    cd $ROOT_DIR/pydensecrf/examples/
    python test_crf.py
    ```
    
4. Use the following commands for final evaluation:

    ```Shell
    cd $ROOT_DIR/examples/saliency/
    python eval_sal.py
    ```
