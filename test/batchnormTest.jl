using Test
include("../src/multiLayerNetExtend.jl")
include("../utils/getMNIST.jl")

function testBatchnorm()
    xTrain, tTrain, xTest, tTest = getMNISTpy()
    # 学習データを削減
    xTrain = xTrain[1:1000, :]
    tTrain = tTrain[1:1000]
    # println(size(xTrain)) #(1000, 784)
    # println(size(tTrain)) #(1000)
    maxEpochs = 20
    trainSize = size(xTrain)[1]
    batchSize  =100
    learningLate = 0.01
end


testBatchnorm()