# SNN_Calibration
**Pytorch Implementation of Spiking Neural Networks Calibration, ICML 2021**

Feature Comparison of SNN calibration:

| Features                | SNN Direct Training                | ANN-SNN Conversion | SNN Calibration |
| ----------------------- | ---------------------------------- | ------------------ | --------------- |
| Accuract ($T<100$)      | High                               | Low                | High            |
| Scalability to ImageNet | Tiny                               | Large              | Large           |
| Training Speed          | Slow                               | Fast               | Fast            |
| # Required Data         | Full-set <br />(1.2M For ImageNet) | ~1000              | ~1000           |
| Inference Speed         | Fast                               | Slow               | Fast            |



### Requirements

Pytorch 1.8

For ImageNet experiments, please be sure that you can initialize distributed environments

For CIFAR experiments, one GPU would suffice. 



### Pre-training ANN on CIFAR10&100

Train an ANN model with `main_train.py`

`python CIFAR/main_train.py --dataset CIFAR10 --arch VGG16 --dpath PATH/TO/DATA --usebn `

Pre-trained results:

| Dataset  | Model     | Random Seed | Accuracy |
| -------- | --------- | ----------- | -------- |
| CIFAR10  | VGG16     | 1000        | 95.76    |
| CIFAR10  | ResNet-20 | 1000        | 95.68    |
| CIFAR100 | VGG16     | 1000        | 77.98    |
| CIFAR100 | ResNet-20 | 1000        | 76.52    |



### SNN Calibration on CIFAR10&100

Calibrate an SNN with `main_calibration.py`.

`python CIFAR/main_calibration.py --dataset CIFAR10 --arch VGG16 --T 16 --usebn --calib advanced --dpath PATH/TO/DATA `

`--T` is the time step, `--calib`  is the calibration method, please use *none, light, advanced* for experiments.  

The calibration will run for 5 times, and return the mean accuracy as well as the standard deviation. 

Example results:

| Architecture | Datset   | T    | Random Seed | Calibration | Mean Acc | Std. |
| ------------ | -------- | ---- | ----------- | ----------- | -------- | ---- |
| VGG16        | CIFAR10  | 16   | 1000        | None        | 64.52    | 4.12 |
| VGG16        | CIFAR10  | 16   | 1000        | Light       | 93.30    | 0.08 |
| VGG16        | CIFAR10  | 16   | 1000        | Advanced    | 93.65    | 0.25 |
| ResNet-20    | CIFAR10  | 16   | 1000        | None        | 67.88    | 3.63 |
| ResNet-20    | CIFAR10  | 16   | 1000        | Light       | 93.89    | 0.20 |
| ResNet-20    | CIFAR10  | 16   | 1000        | Advanced    |          |      |
| VGG16        | CIFAR100 | 16   | 1000        | None        | 2.69     | 0.76 |
| VGG16        | CIFAR100 | 16   | 1000        | Light       | 65.26    | 0.99 |
| VGG16        | CIFAR100 | 16   | 1000        | Advanced    |          |      |
| ResNet-20    | CIFAR100 | 16   | 1000        | None        | 39.27    | 2.85 |
| ResNet-20    | CIFAR100 | 16   | 1000        | Light       | 73.89    | 0.15 |
| ResNet-20    | CIFAR100 | 16   | 1000        | Advanced    |          |      |



### Pre-training ANN on ImageNet

To be updaed



