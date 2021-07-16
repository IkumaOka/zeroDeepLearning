using DataStructures

include("../utils/Affine.jl")
include("../utils/ReLU.jl")
include("../utils/softmaxWithLoss.jl")
include("../utils/errorFunctions.jl")
include("../utils/functions.jl")
include("../utils/gradient.jl")

mutable struct twoLayerNet
    layers::Dict
    params::Dict
    lastLayer
    twoLayerNet() = new()
end

function twoLayerNet(inputSize, hiddenSize, outputSize, weightInitStd=0.01)
    self = twoLayerNet()
    self.params = Dict{}()
    self.params["W1"] = weightInitStd .* randn(inputSize, hiddenSize)
    self.params["b1"] = zeros(1, hiddenSize)
    self.params["W2"] = weightInitStd .* randn(hiddenSize, outputSize)
    self.params["b2"] = zeros(1, outputSize)

    # レイヤの生成
    self.layers = OrderedDict()
    self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
    self.layers["Relu1"] = ReLU(nothing)
    self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

    self.lastLayer = SoftmaxWithLoss(nothing, nothing, nothing)
    return self
end

function predict(self::twoLayerNet, x)
    for layer in values(self.layers)
        x = forward(layer, x)
    end
    return x
end

# x: 入力データ, t: 教師データ
function loss(self::twoLayerNet, x, t)
    y = predict(self, x)
    return forward(self.lastLayer, y, t)
end

function accuracy(self::twoLayerNet, x, t)
    y = predict(self, x)
    y = argmax(y, dims=2)
    # t = argmax(t, dims=2)
    if ndims(a) != 2
        t = argmax(t, dims=2)
    end
    accuracy = sum(y .== t) / size(x)[1]
    return accuracy
end

# x: 入力データ, t: 教師データ
function numericalGradient(self::twoLayerNet, x, t)
    lossW = W -> loss(self, x, t)
    grads = Dict{}()
    grads["W1"] = numericalGradient(lossW, self.params["W1"])
    grads["b1"] = numericalGradient(lossW, self.params["b1"])
    grads["W2"] = numericalGradient(lossW, self.params["W2"])
    grads["b2"] = numericalGradient(lossW, self.params["b2"])

    return grads
end

function gradient(self, x, t)
    # forward
    loss(self, x, t)

    # backward
    dout = 1
    dout = backward(self.lastLayer, dout)

    layers = [t for t in values(self.layers)]
    reverse!(layers)
    for layer in layers
        dout = backward(layer, dout)
    end

    # 設定
    grads = Dict{}()
    grads["W1"] = self.layers["Affine1"].dW
    grads["b1"] = self.layers["Affine1"].db
    grads["W2"] = self.layers["Affine2"].dW
    grads["b2"] = self.layers["Affine2"].db

    return grads
end
