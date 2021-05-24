using Test

include("../src/functions.jl")

function testStep()
    @test step([3, 0, 4])              == [1, 0, 1]
    @test step([-1, -1.5, 1.55555, 1]) == [0, 0, 1, 1]
end

function testSigmoid()
    @test sigmoid([-1.0, 1.0, 2.0]) ≈ [0.2689414213699951, 0.7310585786300049, 0.8807970779778823] atol=0.00001
end

testStep()
testSigmoid()