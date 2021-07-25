using Test

include("../../utils/optims/momentum.jl")

function momentumTest()
    params = Dict{}()
    params["W1"] = [1.1 1.2 1.3; 1.4 1.5 1.6]
    params["W2"] = [1.7 1.8 1.9; 2.0 2.1 2.2]
    grads = Dict{}()
    grads["W1"] = [0.5 0.6 0.7; 0.8 0.9 1.0]
    grads["W2"] = [0.1 0.2 0.3; 0.4 0.5 0.6]
    momentum = Momentum(0.1, 0.9)
    update(momentum, params, grads)

    # 最初の更新はself.vが0で初期化されているため、αvの項が0になり結果はsgdと同じ。
    @test params["W1"] ≈ [1.05 1.14 1.23; 1.32 1.41 1.5] atol=10e-5
    @test params["W2"] ≈ [1.69 1.78 1.87; 1.96 2.05 2.14] atol=10e-5
  
    # 2回目の更新
    update(momentum, params, grads)
    @test params["W1"] ≈ [0.955 1.026 1.097; 1.168 1.239 1.31] atol=10e-5
end

momentumTest()