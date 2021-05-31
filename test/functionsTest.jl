using Test

include("../utils/functions.jl")

function testStep()
    @test step([3, 0, 4])              == [1, 0, 1]
    @test step([-1, -1.5, 1.55555, 1]) == [0, 0, 1, 1]
end

function testSigmoid()
    @test sigmoid([-1.0, 1.0, 2.0]) ≈ [0.2689414213699951, 0.7310585786300049, 0.8807970779778823] atol=0.00001
end

function testRelu()
    @test relu(3)    == 3
    @test relu(-1.5) == 0
    @test relu(0)    == 0
end

function testIdentity()
    @test identity(5)    == 5
    @test identity(-4.0) == -4.0
    @test identity([1 2 3]) == [1 2 3]
end

function testSoftmax()
    @test softmax([0.3 2.9 4.0]) ≈ [0.018211273295547534 0.2451918129350739 0.7365969137693785] atol=0.00001
end

testStep()
testSigmoid()
testRelu()
testIdentity()
testSoftmax()