# GroundeR
This repository contains implementation for *Grounding of Textual Phrases in Images by Reconstruction* in [ECCV 2016](https://arxiv.org/pdf/1511.03745.pdf).

## Setup

*Note*: Please read the feature representation files in ```feature``` and ```annotation``` directories before using the code.

**Platform:** Tensorflow-1.0.1 (python 2.7)<br/>
**Visual features:** We use [Faster-RCNN](https://github.com/endernewton/tf-faster-rcnn) pre-trained on PASCAL 2012 VOC for [Flickr30K Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/), and pre-trained on ImageNet for [Referit Game](http://tamaraberg.com/referitgame/). Please put visual features in the ```feature``` directory (More details can be seen in the [```README.md```](./feature/README.md) in this directory). (Fine-tuned features can achieve better performance, which are available in this [repository](https://github.com/kanchen-usc/QRC-Net).<br/>
**Sentence features:** We encode one-hot vector for each query, as well as the annotation for each query and image pair. Please put the encoded features in the ```annotation``` directory (More details are provided in the [```README.md```](./annotation/README.md) in this directory).<br/>
**File list:** We generate a file list for each image in the Flickr30K Entities. If you would like to train and test on other dataset (e.g. [Referit Game](http://tamaraberg.com/referitgame/)), please follow the similar format in the ```flickr_train_val.lst``` and ```flickr_test.lst```.<br/>
**Hyper parameters:** Please check the ```Config``` class in the ```train.py```.

## Training & Test

We implement both supervised and unsupervised scenarios of GroundeR model.
### Supervised Model
For training, please enter the root folder of ```GroundeR```, then type
```
$ python train_supervise.py -m [Model Name] -g [GPU ID]
```

For testing, please enter the root folder of ```GroundeR```, then type
```
$ python evaluate_supervise.py -m [Model Name] -g [GPU ID] --restore_id [Restore epoch ID]
```
Make sure the model name entered for evaluation is the same as the model name in training, and the epoch id exists.

### Unsupervised Model
The implementation of unsupervised model of GroundeR is a little different from the [paper](https://arxiv.org/pdf/1511.03745.pdf): In Equation 5, original GroundeR adopts a ```softmax``` function to calculate attention weights, while we adopt a ```relu``` function to generate these weights. We observe a performance drop by using ```softmax``` function. To try original GroundeR model, please uncomment line 96 and comment line 97 in ```model_unsupervise.py```.<br/>
For training, please enter the root folder of ```GroundeR```, then type
```
$ python train_unsupervise.py -m [Model Name] -g [GPU ID]
```

For testing, please enter the root folder of ```GroundeR```, then type
```
$ python evaluate_unsupervise.py -m [Model Name] -g [GPU ID] --restore_id [Restore epoch ID]
```
Make sure the model name entered for evaluation is the same as the model name in training, and the epoch id exists.