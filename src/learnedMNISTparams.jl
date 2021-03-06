using JLD2

include("../utils/functions.jl")

function initNetwork()
    network = load("../params/sampleWeight.jld")
    return network
end

function predict(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = x' * W1 .+ b1'
    z1 = sigmoid(a1)
    a2 = (z1 * W2) .+ b2'
    z2 = sigmoid(a2)
    a3 = (z2 * W3) .+ b3'
    y = softmax(a3)
end
