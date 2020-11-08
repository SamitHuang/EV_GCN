# Edge Variational Graph Convolutional Networks for Disease Prediction

## About
This is a Pytorch implementation of EV-GCN described in [Edge-variational Graph Convolutional Networks for Uncertainty-aware Disease Prediction](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_55) (MICCAI 2020) by Yongxiang Huang and Albert C.S. Chung.  

## Prerequisites
- `Python 3.7.4+`
- `Pytorch 1.4.0`
- `torch-geometric `
- `scikit-learn`
- `NumPy 1.16.2`

Ensure Pytorch 1.4.0 is installed before installing torch-geometric. 

This code has been tested using `Pytorch` on a GTX1080TI GPU.

## Training
```
python train_eval_evgcn.py --train=1
```
To get a detailed description for available arguments, please run
```
python train_eval_evgcn.py --help
```
To download the used dataset, please run the following script in the `data` folder: 
```
python fetch_data.py 
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader.py` accordingly, then run `python train_eval_evgcn.py --train=1`  

## Inference and Evaluation
```
python train_eval_evgcn.py --train=0
```

## Reference 
If you find this code useful in your work, please cite:
```
@inproceedings{huang2020edge,
  title={Edge-Variational Graph Convolutional Networks for Uncertainty-Aware Disease Prediction},
  author={Huang, Yongxiang and Chung, Albert CS},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={562--572},
  year={2020},
  organization={Springer}
}
```


