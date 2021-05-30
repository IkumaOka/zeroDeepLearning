using PyCall
include("./learnedMNISTparams.jl")

# shape of x = 28 * 28 * 10000
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
network = initNetwork()
x = reshape(x, 10000, 784)

batchSize = 100
accuracyCnt = 0
# println(length(x[:, 1])) # 10000
for i in 1:100:(length(x[:, 1]) + 1 - batchSize)
    Xbatch = x[i:i+batchSize-1, :]
    # println(length(Xbatch[:, 1])) # 100
    # println(length(Xbatch[1, :])) # 784
    # Xi = reshape(Xbatch, length(Xbatch[:, 1]), batchSize)
    # println(Xi)
    yBatch = predict(network, Xbatch')
    p = mapslices(argmax, yBatch, dims=2)
    global accuracyCnt += sum(p.-1 .== t[i:i+batchSize-1])
end
print("Accuracy:", accuracyCnt / length(x[:, 1]))