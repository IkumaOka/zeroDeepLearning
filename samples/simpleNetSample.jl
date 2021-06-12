include("../src/gradientSimpleNet.jl")
include("../utils/gradient.jl")

#W = randn(2, 3)

# 本の例
W = [0.47355232 0.9977393 0.84668094;
     0.85557411 0.03563661 0.69422093]

net = simpleNet(W)
x = [0.6 0.9]
p = predict(net, x)
println(p)
println(argmax(p)[2])

t = [0 0 1]
println(loss(net, x, t))

function f(W)
     return loss(net, x, t)
end

dW = numericalGradient(f, net.W)
println(dW)