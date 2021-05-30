using PyCall

py"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import pickle
from datasets.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

x, t = get_data()
"""
x = py"x"
t = py"t"
print(x)