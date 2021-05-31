using Test
include("../utils/errorFunctions.jl")

function testmeanSquared()
    t = [0 0 1 0 0 0 0 0 0 0]
    y = [0.1 0.05 0.6 0.0 0.05 0.1 0.0 0.1 0.0 0.0]
    @test meanSquared(y, t) ≈ 0.0975 atol=0.00001

    y = [0.1 0.05 0.1 0.0 0.05 0.1 0.0 0.6 0.0 0.0]
    @test meanSquared(y, t) ≈0.5975 atol=0.00001
end

testmeanSquared()