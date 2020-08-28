# RGCN
This is an implementation of "Robust Graph Convolutional Networks Against Adversarial Attacks". 

### Usage
##### Example Usage
```
python src/train.py --dataset cora
```
##### Full Command List
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
