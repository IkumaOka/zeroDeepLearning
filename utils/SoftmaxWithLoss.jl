include("./errorFunctions.jl")
include("./functions.jl")

mutable struct SoftmaxWithLoss
    loss
    y
    t
end

function forward(self::SoftmaxWithLoss, x::Array, t::Array)
    self.t = t
    self.y = softmax(x)
    self.loss = crossEntropy(self.y, self.t)

    return self.loss
end

function backward(self::SoftmaxWithLoss, dout=1)
    batchSize = size(self.t)[1]
    dx = (self.y .- self.t) ./ batchSize
    
    return dx
end
