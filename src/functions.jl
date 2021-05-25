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
    expa = exp.(x)
    sumExp = sum(expa)
    return exp.(x) ./ sum(expa)
end