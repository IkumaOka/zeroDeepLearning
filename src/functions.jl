function step(x)
    return x .> 0
end

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end