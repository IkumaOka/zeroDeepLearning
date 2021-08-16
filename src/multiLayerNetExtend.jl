using DataStructures

include("../utils/Affine.jl")
include("../utils/ReLU.jl")
include("../utils/SoftmaxWithLoss.jl")
include("../utils/errorFunctions.jl")
include("../utils/functions.jl")
include("../utils/gradient.jl")

mutable struct MultiLayerNetExtend
    inputSize::Array{Int64, 2}
    outputSize::Array{Int64, 2}
    hiddenSizeList::Array{Int64, 2}
    hiddenLayerNum::Int64
    useDropout::Bool
    weightDecayLambda::Float64
    useBatchnorm::Bool
    params::Dict
    MultiLayerNetExtend() = new()
end

function MultiLayerNetExtend(inputSize, hiddenSizeList, outputSize,
                             activation="relu", weightInitStd="relu", weightDecayLambda=0,
                             useDropout=false, dropoutRation=0.5, useBatchnorm=false)
    self = MultiLayerNetExtend()
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.hiddenSizeList = hiddenSizeList
    self.hiddenLayerNum = length(hiddenSizeList)
    self.useDropout = useDropout
    self.weightDecayLambda = weightDecayLambda
    self.useBatchnorm = useBatchnorm
    self.params = Dict{}()
    # 重みの初期化
    __initWeight(self, weightInitStd)
    # レイヤの生成
    activationLayer = Dict("sigmoid" => Sigmoid, "relu" => ReLU)
end