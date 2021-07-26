using Test

include("../../utils/optims/adagrad.jl")

function AdaGradTest()
    params = Dict{}()
    params["W1"] = [1.1 1.2 1.3; 1.4 1.5 1.6]
    grads = Dict{}()
    grads["W1"] = [0.5 0.6 0.7; 0.8 0.9 1.0]
    adagrad = AdaGrad(0.1)
    update(adagrad, params, grads)

    @test params["W1"] ≈ [1.0 1.1 1.2; 1.3 1.4 1.5] atol=10e-5
    
    # 2回目の更新
    update(adagrad, params, grads)
    @test params["W1"] ≈ [ 0.929289 1.02929 1.12929; 1.22929 1.32929 1.42929] atol=10e-5
end

AdaGradTest()