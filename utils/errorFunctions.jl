using StatsBase

function meanSquared(y, t)
    return 0.5 .* sum((y .- t) .^ 2)
end

function crossEntropy(y, t)
    delta = 1e-7
    return -sum(t .* log.(y .+ delta)) / size(y)[1]
end