using Test
include("../utils/errorFunctions.jl")

function testMeanSquared()
    t = [0 0 1 0 0 0 0 0 0 0]
    y = [0.1 0.05 0.6 0.0 0.05 0.1 0.0 0.1 0.0 0.0]
    @test meanSquared(y, t) ≈ 0.0975 atol=0.00001

    y = [0.1 0.05 0.1 0.0 0.05 0.1 0.0 0.6 0.0 0.0]
    @test meanSquared(y, t) ≈ 0.5975 atol=0.00001
end

function testCrossEntropy()
    t = [0 0 1 0 0 0 0 0 0 0]
    y = [0.1 0.05 0.6 0.0 0.05 0.1 0.0 0.1 0.0 0.0]
    @test crossEntropy(y, t) ≈ 0.51082545709933802 atol=0.00001

    y = [0.1 0.05 0.1 0.0 0.05 0.1 0.0 0.6 0.0 0.0]
    @test crossEntropy(y, t) ≈ 2.3025840929945458 atol=0.00001
end


testMeanSquared()
testCrossEntropy()