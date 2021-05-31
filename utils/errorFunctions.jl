function meanSquared(y, t)
    return 0.5 .* sum((y .- t) .^ 2)
end