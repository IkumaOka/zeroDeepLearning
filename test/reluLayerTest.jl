using Test
include("../utils/ReLULayer.jl")

function TestRelu()
    x = [1.0 -0.5; -2.0 3.0]
    relu = ReLULayer(nothing)
    f = forward(relu, x)
    b = backward(relu, x)
    @test f == [1.0 0; 0 3.0]
    @test b == [1.0 0; 0 3.0]
end

TestRelu()