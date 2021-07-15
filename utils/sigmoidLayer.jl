mutable struct sigmoidLayer
    out::Array
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
 
