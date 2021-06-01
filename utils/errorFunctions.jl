using StatsBase

function meanSquared(y, t)
    return 0.5 .* sum((y .- t) .^ 2)
end

# 後日Batch版の対応する（p.94~）
function crossEntropy(y, t)
    delta = 1e-7
    return -sum(t .* log.(y .+ delta))
end