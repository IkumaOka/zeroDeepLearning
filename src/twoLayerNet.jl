include("../utils/errorFunctions.jl")
include("../utils/functions.jl")
include("../utils/gradient.jl")

mutable struct twoLayerNet
    params::Dict
    twoLayerNet() = new()
end

function twoLayerNet(inputSize, hiddenSize, outputSize, weightInitStd=0.01)
    self = twoLayerNet()
    self.params = Dict{}()
    self.params["W1"] = weightInitStd .* randn(inputSize, hiddenSize)
    self.params["b1"] = zeros(1, hiddenSize)
    self.params["W2"] = weightInitStd .* randn(hiddenSize, outputSize)
    self.params["b2"] = zeros(1, outputSize)
    return self
end

function predict(self::twoLayerNet, x)
    W1, W2 = self.params["W1"], self.params["W2"]
    b1, b2 = self.params["b1"], self.params["b2"]
    a1 = x * W1 .+ b1
    z1 = sigmoid(a1)
    a2 = (z1 * W2) .+ b2
    y = sigmoid(a2)
    return y
end

# x: 入力データ, t: 教師データ
function loss(self::twoLayerNet, x, t)
    y = predict(self, x)
    return crossEntropy(y, t)
end

function accuracy(self::twoLayerNet, x, t)
    y = predict(self, x)
    y = argmax(y, dims=2)
    t = argmax(t, dims=2)
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