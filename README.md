# RobustGCN
This is a sample implementation of "[Robust Graph Convolutional Networks Against Adversarial Attacks](https://zw-zhang.github.io/files/2019_KDD_RGCN.pdf)", KDD 2019. 

### Requirements
```
tensorflow >= 1.12
numpy >= 1.14.2
scipy >= 1.1.0
networkx >= 2.0.0
gcn (note that you need to follow https://github.com/tkipf/gcn to correctly install gcn instead of using pip)
```
### Example Usage
```
python src/train.py --dataset cora
```
### Full Command List
```
optional arguments:
--dataset  Dataset string.
--learning_rate  Initial learning rate.
--epochs  Number of epochs to train.
--hidden  Number of units in hidden layer.
--dropout Dropout rate (1 - keep probability).
--para_var Parameter of variance-based attention.
--para_kl Parameter of kl regularization.
--para_l2 Parameter for l2 loss.
--early_stopping Tolerance for early stopping (# of epochs).
```
### Cite
If you find this code useful, please cite our paper:
```
@inproceedings{zhu2019robust,
  title={Robust graph convolutional networks against adversarial attacks},
  author={Zhu, Dingyuan and Zhang, Ziwei and Cui, Peng and Zhu, Wenwu},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1399--1407},
  year={2019}
}
```
### Acknowledgement
Our code is adapted from the Tensorflow implementation of GCN by Thomas Kipf (https://github.com/tkipf/gcn). 
