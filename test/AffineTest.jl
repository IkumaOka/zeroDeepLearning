using Test
include("../utils/Affine.jl")

function TestAffine()
    x = [1 2; 3 4; 5 6]
    W = [0.1 0.2 0.3; 0.4 0.5 0.6]
    b = [10 11 12]

    affine = Affine(W, b)
    f = forward(affine, x)

    @test f ≈ [10.9 12.2 13.5; 11.9 13.6 15.3; 12.9 15.0 17.1] atol=0.00001
end

TestAffine()