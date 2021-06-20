include("../src/twoLayerNet.jl")
include("../utils/gradient.jl")

net = twoLayerNet(784, 100, 10)
#println(size(net.params["W1"])) # (784, 100)
#println(size(net.params["b1"])) # (100,)
#println(size(net.params["W2"])) # (100, 10)
#println(size(net.params["b2"])) # (10,)

x = rand(100, 784) # ダミー入力データ（100枚分）
t = rand(100, 10) # ダミーの正解ラベル（100枚分）

grads = numericalGradient(net, x, t)

#println(size(grads["W1"])) # (784, 100)
#println(size(grads["b1"])) # (1, 100)
#println(size(grads["W2"])) # (100, 10)
#println(size(grads["b2"])) # (1, 10)