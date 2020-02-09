## [A Simple Saliency Detection Approach via Automatic Top-Down Feature Fusion](https://www.sciencedirect.com/science/article/abs/pii/S0925231220300709) 

### Introduction

It is widely accepted that the top sides of convolutional neural networks (CNNs) convey high-level semantic features, and the bottom sides contai low-level details. Therefore, most of recent salient object detection methods aim at designing effective fusion strategies for side-output features. Although significant progress has been achieved in this direction, the network architectures become more and more complex, which will make the future improvement difficult and heavily engineered. Moreover, the manually designed fusion strategies would be sub-optimal due to the large search space of possible solutions. To address above problems, we propose an Automatic Top-Down Fusion (ATDF) method, in which the global information at the top sides are flowed into bottom sides to guide the learning of low layers. We design a novel valve module and add it at each side to control the coarse semantic information flowed into a specific bottom side. Through these valve modules, each bottom side at the top-down pathway is expected to receive necessary top information. We also design a generator to improve the prediction capability of fused deep features for saliency detection. We perform extensive experiments to demonstrate that ATDF is simple yet effective and thus opens a new path for saliency detection.

### Citations

If you are using the code/model/data provided here in a publication, please consider citing our papers:

    @article{qiu2020simple,
      title={A Simple Saliency Detection Approach via Automatic Top-Down Feature Fusion},
      author={Qiu, Yu and Liu, Yun and Yang, Hui and Xu, Jing},
      journal={Neurocomputing},
      year={2020},
      publisher={Elsevier}
    }
    
    @inproceedings{qiu2019revisiting,
      title={Revisiting Multi-Level Feature Fusion: A Simple Yet Effective Network for Salient Object Detection},
      author={Qiu, Yu and Liu, Yun and Ma, Xiaoxu and Liu, Lei and Gao, Hongcan and Xu, Jing},
      booktitle={IEEE International Conference on Image Processing},
      pages={4010--4014},
      year={2019}
    }
    
### Pre-computed saliency maps

The pre-computed saliency maps for six datasets, including DUT-OMRON, DUTS, ECSSD, HKU-IS, SOD, and THUR15K, can be found in the folder of `SaliencyMaps`.

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
   Here, `$ROOT_DIR/saliency_data/DUTS-TR/*.jpg` is the path of training images, and `$ROOT_DIR/saliency_data/DUTS-TR/*.png` is the path of the saliency ground truth. Maybe you need to change the path to your own data.


4. Download the pretrained vgg16 model [here](http://mftp.mmcheng.net/liuyun/rcf/model/5stage-vgg.caffemodel).

5. Start the training process by running the following commands:

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
   Here, `$ROOT_DIR/saliency_data/ECSSD/*.jpg` is the path of test images.

2. Change the paths for the data and model in `test_sal.py`, and start the test process by running the following commands:

    ```Shell
    cd $ROOT_DIR/examples/saliency/
    python test_sal.py
    ```
    
3. Follow [this website](https://github.com/Andrew-Qibin/dss_crf) to install CRF package, and perform CRF by running the following commands:
    
    ```Shell
    cd $ROOT_DIR/pydensecrf/examples/
    python test_crf.py
    ```
    
4. Use the following commands for final evaluation:

    ```Shell
    cd $ROOT_DIR/examples/saliency/
    python eval_sal.py
    ```
