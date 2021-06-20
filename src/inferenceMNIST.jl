using PyCall
include("./learnedMNISTparams.jl")


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

# shape of x = 28 * 28 * 10000
x = py"x"
t = py"t"

network = initNetwork()
x = reshape(x, 10000, 784)
# println(x[10000, :]) # 784
accuracyCnt = 0
for i in 1:length(x[:, 1])
    # Xi =  reshape(x[:, :, i]', 1, length(x[:, :, i])) # (1, 784)
    y = predict(network, x[i, :])
    p = findmax(y)[2][2]
    # tはpythonから引っ張ってきてるため、indexに0が含まれる。そのため、p-1をしなければならない。
    if p-1 == t[i]
        global accuracyCnt += 1
    end
end
print("Accuracy:", accuracyCnt / length(x[:, 1]))