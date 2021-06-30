#MNISTデータをtwoLayerNetでミニバッチ学習させる(本4.5.2節p.117~)
using StatsBase

include("../src/twoLayerNet.jl")
include("../utils/getMNIST.jl")

xTrain, tTrain, xTest, tTest = getMNISTpy()

#println(size(xTrain)) #(60000, 784)
#println(size(tTrain)) #(60000,)

trainLossList = []

# ハイパーパラメータ
itersNum = 10000
trainSize = size(xTrain)[1]
batchSize = 100
learningRate = 0.1

network = twoLayerNet(784, 50, 10)

println(size(xTrain[60000, :]))

for i in 1:itersNum
    # ミニバッチの取得
    batchMask = StatsBase.sample(1:trainSize, batchSize)
    xBatch = xTrain[batchMask]
    println(xBatch)
    println(size(xBatch))
    tBatch = tTrain[batchMask]

    # 勾配の計算
    grad = numericalGradient(network, xBatch, tBatch)
    # grad = gradient(network, Xbatch, tBatch) # 高速版

    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2")
        network.params[key] .-= learningRate .* grad[key]
    end

    # 学習経過の記録
    lossVal = loss(network, xBatch, tBatch)
    push!(trainLossList, lossVal)
end


