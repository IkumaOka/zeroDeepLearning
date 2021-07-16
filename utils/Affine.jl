mutable struct Affine
    Affine() = new()
    W::Array
    b::Array
    x
    dW
    db
end

function Affine(W::Array, b::Array)
    self = Affine()
    self.W = W
    self.b = b
    self.x = nothing
    self.dW = nothing
    self.db = nothing

    return self
end

function forward(self::Affine, x::Array)
    self.x = x
    out = self.x * self.W .+ self.b

    return out
end

function backward(self::Affine, dout)
    dx = dout * self.W'
    self.dW = self.x' * dout
    self.db = sum(dout, dims=1)

    return dx
end
