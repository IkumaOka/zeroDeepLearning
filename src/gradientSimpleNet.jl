include("../utils/functions.jl")

mutable struct simpleNet
    W::Array
end

function simpleNet()
    self = simpleNet()
    self.W = randn(2, 3)

    return self
end

function predict(self::simpleNet, x)
    return x * self.W
end

function loss(self::simpleNet, x, t)
    z = self.predict(x)
    y = softmax(z)
    loss = crossEntropy(y, t)

    return loss
end