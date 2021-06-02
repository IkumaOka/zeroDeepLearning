using Test

include("../utils/gradient.jl")

function function1(x)
    return 0.01 .* x.^2 .+ 0.1 .* x
end

function function2(x)
    return x[1].^2 .+ x[2] .^2
end

function testNumericalDiff()
    # 内包表記以外に作り方あるか調べる
    x =  [n for n = 0.0:0.1:20]
    @test numericalDiff(function1, 5)  ≈ 0.1999999999990898 atol=0.00001
    @test numericalDiff(function1, 10) ≈ 0.2999999999986347 atol=0.00001
end

function testNumericalGradient()
    @test numericalGradient(function2, [3.0 4.0]) ≈ [6.00000000000378 7.999999999999119] atol=0.00001
    @test numericalGradient(function2, [0.0 2.0]) ≈ [0.0 4.000000000004] atol=0.00001
    @test numericalGradient(function2, [3.0 0.0]) ≈ [6.000000000012662 0.0] atol=0.00001
end

function testGradientDescent()
    initX = [-3.0 4.0]
    @test gradientDescent(function2, initX, 1e-10, 100) ≈ [-2.999999939999995 3.9999999199999934] atol=0.00001
end




testNumericalDiff()
testNumericalGradient()
testGradientDescent()