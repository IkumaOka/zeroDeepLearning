mutable struct Sigmoid
    out::Array
end

function forward(self::Sigmoid, x)
    out = 1 ./ (1 .+ exp.(-x))
    self.out = out

    return out
end

function backward(self::Sigmoid, dout)
    dx = dout .* (1.0 .- self.out) .* self.out

    return dx
end
 
