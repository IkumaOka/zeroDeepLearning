using DataStructures

include("../utils/optims/adagrad.jl")
include("../utils/optims/adam.jl")
include("../utils/optims/momentum.jl")
include("../utils/optims/sgd.jl")

function f(x::Float64, y::Float64)
    return x^2 / 20.0 + y^2
end

# fを微分した関数
function df(x::Float64, y::Float64)
    return x / 10.0, 2.0 * y
end

initPos = (-7.0, 2.0)
params = Dict{}()
params["x"], params["y"] = initPos[1], initPos[2]
grads = Dict{}()

grads["x"], grads["y"] = 0, 0

optimizers = OrderedDict()

optimizers["SGD"] = SGD(0.95)
optimizers["Momentum"] = Momentum(0.1, 0.9)
optimizers["Adam"] = Adam(0.3, 0.9, 0.999)
optimizers["AdaGrad"] = AdaGrad(1.5)

idx = 1

for key in keys(optimizers)
    optimizer = optimizers[key]
    xHistory = []
    yHistory = []

    params["x"], params["y"] = initPos[1], initPos[2]

    for i in 1:30
        push!(xHistory, params["x"])
        push!(yHistory, params["y"])

        grads["x"], grads["y"] = df(params["x"], params["y"])
        update(optimizer, params, grads)
    end

    x = [n for n = -10.0:0.01:10.0]
    y = [n for n = -5.0:0.01:5.0]

    

end
