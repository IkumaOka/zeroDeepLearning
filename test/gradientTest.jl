using Test

include("../utils/gradient.jl")

function function1(x)
    return 0.01 .* x.^2 .+ 0.1 .* x
end

function testNumericalDiff()
    # 内包表記以外に作り方あるか調べる
    x =  [n for n = 0.0:0.1:20]
    @test numericalDiff(function1, 5)  ≈ 0.1999999999990898 atol=0.00001
    @test numericalDiff(function1, 10) ≈ 0.2999999999986347 atol=0.00001
end

testNumericalDiff()