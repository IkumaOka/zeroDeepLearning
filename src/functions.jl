function step(x)
    return x .> 0
end

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x)
    return max(0, x)
end

# 恒等関数
function identity(x)
    return x
end

function softmax(x)
    # findmax(x)[1]を引く理由はオーバーフロー対策
    expa = exp.(x .- findmax(x)[1])
    return expa ./ sum(expa)
end