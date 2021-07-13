using Test
include("../utils/sigmoidLayer.jl")

function TestSigmoid()
    x = [1.0 -0.5; -2.0 3.0]
    sigmoid = sigmoidLayer(nothing)
    f = forward(sigmoid, x)
    b = backward(sigmoid, x)
    @test f ≈ [0.7310585786300049 0.3775406687981454; 0.11920292202211755 0.9525741268224334] atol=0.0001
    @test b ≈ [0.19661193324148185 -0.11750185610079725; -0.209987170807013 0.13552997919273602] atol=0.0001
end

TestSigmoid()