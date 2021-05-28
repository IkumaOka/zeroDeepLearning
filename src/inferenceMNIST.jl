using MLDatasets
using JLD2

include("../utils/functions.jl")


function getData()
    trainX, trainY = MNIST.traindata()
    testX,  testY  = MNIST.testdata()
    return testX, testY
end

function initNetwork()
    network = load("../params/sampleWeight.jld")
    return network
end

function predict(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = x * W1 + b1'
    z1 = sigmoid(a1)
    a2 = (z1 * W2) + b2'
    z2 = sigmoid(a2)
    a3 = (z2 * W3) + b3'
    y = softmax(a3)
end

# shape of x = 28 * 28 * 10000
x, t = getData()
network = initNetwork()

accuracyCnt = 0
for i in 1:length(x[28, 28, :])
    Xi =  reshape(x[:, :, i]', length(x[:, :, i]), 1)
    y = predict(network, Xi')
    println(y)
    p = findmax(y)[2][2]
    println(p)
    if p == t[i]
        global accuracyCnt += 1
    end
end
print("Accuracy:", accuracyCnt / length(x[28, 28, :]))