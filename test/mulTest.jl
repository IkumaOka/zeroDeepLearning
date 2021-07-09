using Test
include("../utils/mulLayer.jl")

# buy_apple.py(p.138と同じ)
function TestMul()
    apple = 100
    appleNum = 2
    tax = 1.1

    # layer
    # MulLayerの引数はなんでもよい（ダミーで渡してるだけ、ここら辺のJuliaの仕様がよくわからない）
    mulAppleLayer = MulLayer(apple)
    mulTaxLayer = MulLayer(apple)

    # forward
    applePrice = forward(mulAppleLayer, apple, appleNum)
    price = forward(mulTaxLayer, applePrice, tax)
    @test price ≈ 220 atol=0.00001

    # backward
    dprice = 1
    dApplePrice, dTax = backward(mulTaxLayer, dprice)
    dApple, dAppleNum = backward(mulAppleLayer, dApplePrice)
    @test dApple ≈ 2.2 atol=0.00001
    @test dAppleNum ≈ 110 atol=0.00001
    @test dTax ≈ 200 atol=0.00001
end

TestMul()