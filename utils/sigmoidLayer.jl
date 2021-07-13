mutable struct sigmoidLayer
    # いらないかも。Python実装はあるから一応残す。(p.142)。いらない場合の実装分からん。
    out
end

function forward(self::sigmoidLayer, x)
    out = 1 ./ (1 .+ exp.(-x))
    self.out = out

    return out
end

function backward(self::sigmoidLayer, dout)
    dx = dout .* (1.0 .- self.out) .* self.out

    return dx
end
 
