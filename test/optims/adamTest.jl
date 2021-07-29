using Test

include("../../utils/optims/adam.jl")

function AdamTest()
    params = Dict{}()
    params["W1"] = [1.1 1.2 1.3; 1.4 1.5 1.6]
    grads = Dict{}()
    grads["W1"] = [0.5 0.6 0.7; 0.8 0.9 1.0]
    adam = Adam(0.1, 0.9, 0.999)
    update(adam, params, grads)

    # 2回目の更新
    update(adam, params, grads)
    @test params["W1"] ≈ [0.9 1.0 1.1; 1.2 1.3 1.4] atol=10e-5
end

AdamTest()