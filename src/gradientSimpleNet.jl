include("../utils/errorFunctions.jl")
include("../utils/functions.jl")


mutable struct simpleNet
    W::Array{Float64,2}
    simpleNet() = new()
end

function simpleNet(W::Array{Float64,2})
    self = simpleNet()
    self.W = W
    return self
end

function predict(self::simpleNet, x)
    return x * self.W
end

function loss(self::simpleNet, x, t)
    z = predict(self, x)
    y = softmax(z)
    loss = crossEntropy(y, t)

    return loss
end