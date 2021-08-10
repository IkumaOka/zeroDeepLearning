using Test

# pythonのnp.meshgridをjuliaでどうやるかと、値が合うかのテスト
# juliaではQuantEconのmeshgrid関数でmeshgridを使えそうだけど、使わなくても実装できそう

function f(x::Float64, y::Float64)
    return x^2 / 20.0 + y^2
end

function testMeshgrid()
    # 例1
    x1 = [0.0 1.0 2.0]
    y1 = [0.0 1.0 2.0]
    x1Grid = repeat(x1, outer=(length(y1),1))
    y1Grid = repeat(y1',  outer=(1,length(x1)))
    Z1 = f.(x1Grid, y1Grid)

    # 例2
    x2 = [3.0 8.0 -7.0]
    y2 = [1.0 -5.0 4.0]
    x2Grid = repeat(x2, outer=(length(y2),1))
    y2Grid = repeat(y2',  outer=(1,length(x2)))
    Z2 = f.(x2Grid, y2Grid)

    @test Z1 ≈ [0.0  0.05  0.2; 1.0  1.05  1.2; 4.0  4.05  4.2]   atol=0.00001
    @test Z2 ≈ [1.45 4.2 3.45; 25.45 28.2 27.45; 16.45 19.2 18.45] atol=0.00001
end


testMeshgrid()