# MVE
This is an implementation of the MVE model for multi-view network embedding [(An Attention-based Collaboration Framework for Multi-View Network Representation Learning)](https://arxiv.org/abs/1709.06636). 

## Install
Our codes rely on two external packages, which are the Eigen package and the GSL package.

#### Eigen
The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) package is used for matrix operations. To run our codes, users need to download the Eigen package and modify the package path in the makefile.

#### GSL
The [GSL](https://www.gnu.org/software/gsl/) package is used to generate random numbers. After installing the package, users also need to modify the package path in the makefile. 

## Compile
After installing the two packages and modifying the package paths, users may go to the "mve" folder and use the makefile to compile the codes.

## Data
The MVE model receives a multi-view network and a set of labeled nodes as input. 

Each view of the multi-view network is described by a single file. The files of different views should have the same prefix, and the indices should start from 0 to K-1, where K is the number of views. For example, users may describe a multi-view network with three files, "view_0", "view_1", "view_2", where the prefix is "view_" and the number of views is 3. Each view file contains several lines, with each line representing an edge in that view. The format of each line is: <u> <v> <w>, meaning that there is an edge from node <u> to node <v> and the weight is <w>.

The labeled nodes are listed in another file. This file contains several lines, where each line gives the labels of a node. The format of a line is: <node name> <label name 1> <label name 2>, which starts with the name of the node, followed by the names of different labels the node has.

A toy dataset is provided in the "data/toy/" folder, and we will upload the dataset used in the paper later.

## Running
To run the MVE model, users may directly use the example script (run.sh) we provide. 

## Contact
If you have any questions about the codes, please feel free to contact us.
```
Meng Qu, qumn123@gmail.com
```

## Citation
```
@inproceedings{qu2017attention,
title={An Attention-based Collaboration Framework for Multi-View Network Representation Learning},
author={Qu, Meng and Tang, Jian and Shang, Jingbo and Ren, Xiang and Zhang, Ming and Han, Jiawei},
booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
pages={1767--1776},
year={2017},
organization={ACM}
}
```

