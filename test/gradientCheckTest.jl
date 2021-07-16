# 数値微分と解析的によって求めた勾配の値がほぼ同じになるかのcheck
using StatsBase
using Test
include("../src/twoLayerNet.jl")
include("../utils/getMNIST.jl")

function gradientCheck()
    xTrain, tTrain, xTest, tTest = getMNISTpy()
    network = twoLayerNet(784, 50, 10)

    xBatch = xTrain[1:3, 1:784]
    tBatch = tTrain[1:3]

    gradNumerical = numericalGradient(network, xBatch, tBatch)
    gradBackprop = gradient(network, xBatch, tBatch)

    # 各重みの絶対誤差の平均を求める
    for key in keys(gradNumerical)
        diff = mean(abs.(gradBackprop[key] - gradNumerical[key]))
        println(key, ":", diff)
    end
end

gradientCheck()

