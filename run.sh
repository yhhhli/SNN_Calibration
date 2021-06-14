#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:. GLOG_vmodule=MemcachedClient=-1 
spring.submit run --gpu -n1 --cpus-per-task 4 "python CIFAR/main_train.py --dataset CIFAR10 --arch VGG16 --dpath /mnt/lustre/share/prototype_cifar/cifar10/ --usebn"

