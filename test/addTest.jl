using Test
include("../utils/addLayer.jl")
include("../utils/mulLayer.jl")

# buy_apple.py(p.138と同じ)
function TestAdd()
    apple = 100
    appleNum = 2
    orange = 150
    orangeNum = 3
    tax = 1.1

    # layer
    # addLayerの引数はなんでもよい（ダミーで渡してるだけ、ここら辺のJuliaの仕様がよくわからない）
    mulAppleLayer = MulLayer(apple)
    mulOrangeLayer = MulLayer(apple)
    addAppleOrangeLayer = AddLayer(apple)
    mulTaxLayer = MulLayer(apple)

    # forward
    applePrice = forward(mulAppleLayer, apple, appleNum)
    orangePrice = forward(mulOrangeLayer, orange, orangeNum)
    allPrice = forward(addAppleOrangeLayer, applePrice, orangePrice)
    price = forward(mulTaxLayer, allPrice, tax)

    @test price ≈ 715 atol=0.00001

    # backward
    dprice = 1
    dAllPrice, dTax = backward(mulTaxLayer, dprice)
    dApplePrice, dOrangePrice = backward(addAppleOrangeLayer, dAllPrice)
    dOrange, dOrangeNum = backward(mulOrangeLayer, dOrangePrice)
    dApple, dAppleNum = backward(mulAppleLayer, dApplePrice)
    
    @test dAppleNum ≈ 110 atol=0.00001
    @test dApple ≈ 2.2 atol=0.00001
    @test dOrangeNum ≈ 165 atol=0.00001
    @test dOrange ≈ 3.3 atol=0.00001
    @test dTax ≈ 650 atol=0.00001
end

TestAdd()